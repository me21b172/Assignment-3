
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import wandb_runner
import time
from utils import int2devnagri,devnagri2int, showPlot,timeSince
from model import EOS_TOKEN,SOS_TOKEN

PAD_TOKEN = "<PAD>"


def evaluate_model(encoder, decoder, dataloader, int2devnagri, device, teacher_forcing_prob,beam_width,show_confusion=True, iswandb=False):
    font_path = 'C:/Users/aksha/Downloads/Noto_Sans_Devanagari/NotoSansDevanagari-VariableFont_wdth,wght.ttf'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans Devanagari'
    
    encoder.eval()
    decoder.eval()

    correct_words = 0
    total_words = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for _, _, inputs, targets, input_lengths, _ in dataloader:
            # _, _, input_tensor, target_tensor, input_lengths, target_lengths = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            encoder_outputs, encoder_hidden, encoder_cell = encoder(inputs, input_lengths)
            # encoder_hidden = tuple(h.to(device) for h in encoder_hidden) if isinstance(encoder_hidden, tuple) \
                            # else encoder_hidden.to(device)
            # encoder_cell = encoder_cell.to(device) if encoder_cell is not None else None
            decoder_outputs, _, _, _ = decoder(encoder_outputs, encoder_hidden, encoder_cell,teacher_forcing_prob=teacher_forcing_prob,beam_width=beam_width)
            # print(decoder_outputs)
            # print(f"decoder_outputs shape: {decoder_outputs.shape}")
            if decoder_outputs.dtype == torch.long:
    # beam search: output is token IDs already
                predicted_indices = decoder_outputs
            else:
                # teacher-forcing: output is (batch, seq_len, vocab_size) logits
                predicted_indices = decoder_outputs.argmax(dim=-1)


            for i, (pred_seq, true_seq) in enumerate(zip(predicted_indices, targets)):
                # print(f"Batch {i}: pred_seq shape: {pred_seq.shape if hasattr(pred_seq, 'shape') else 'scalar'}, value: {pred_seq}")
                if pred_seq.dim() == 0:
                    pred_list = [pred_seq.item()] if pred_seq.item() != devnagri2int[PAD_TOKEN] else []
                else:
                    pred_list = [i.item() for i in pred_seq if i.item() != devnagri2int[PAD_TOKEN]]
                true_list = [i.item() for i in true_seq if i.item() != devnagri2int[PAD_TOKEN]]

                pred_str,true_str = [],[]
                for char in pred_list:
                    if char is devnagri2int[EOS_TOKEN]:
                        break
                    if char is not devnagri2int[SOS_TOKEN]:
                        pred_str.append(int2devnagri[char])
                for char in true_list:
                    if char is devnagri2int[EOS_TOKEN]:
                        break
                    if char is not devnagri2int[SOS_TOKEN]:
                        true_str.append(int2devnagri[char])
                pred_str = ''.join(pred_str)
                true_str = ''.join(true_str)
                # print(pred_str)
                # print(true_str)
                if pred_str == true_str:
                    correct_words += 1
                total_words += 1

                min_len = min(len(pred_list), len(true_list))
                y_true.extend(true_list[:min_len])
                y_pred.extend(pred_list[:min_len])

    word_accuracy = correct_words / total_words if total_words > 0 else 0.0
    
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(['Correct', 'Incorrect'], 
                  [correct_words, total_words - correct_words],
                  color=['#4CAF50', '#F44336'])
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,}\n({height/total_words:.1%})',
                ha='center', va='bottom')
    
    ax1.set_title(f'Word Accuracy: {word_accuracy:.2%}\nTotal Words: {total_words:,}', pad=20)
    ax1.set_ylabel('Count', labelpad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    if show_confusion and y_true and y_pred:
        ax2 = fig.add_subplot(1, 2, 2)
        labels = sorted(list(set(y_true + y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        step = max(1, len(labels)//20)
        display_labels = [int2devnagri[label] if i%step==0 else '' 
                         for i, label in enumerate(labels)]
        
        sns.heatmap(cm_normalized, ax=ax2,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Accuracy Percentage', 'shrink': 0.7},
                   xticklabels=display_labels,
                   yticklabels=display_labels,
                   annot=False,
                   square=True)
        
        ax2.set_title('Character Prediction Patterns', pad=20)
        ax2.set_xlabel('Predicted Characters', labelpad=10)
        ax2.set_ylabel('True Characters', labelpad=10)
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax2.get_yticklabels(), rotation=0)
        
        for _, spine in ax2.spines.items():
            spine.set_visible(True)
            spine.set_color('gray')
        if iswandb:
            wandb_runner.log({"confusion_matrix": wandb_runner.Image(fig)})
    plt.tight_layout()
    
    plt.show()
    plt.savefig("acc.png")

    encoder.train()
    decoder.train()
    return word_accuracy

def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer):
    total_loss = 0
    for data in dataloader:
        _, _, input_tensor, target_tensor, input_lengths, target_lengths = data
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder forward
        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor, input_lengths)
        # Decoder forward (use target tensor without last token)
        decoder_outputs, _, _, attention = decoder(
            encoder_outputs, encoder_hidden, encoder_cell,
            target_tensor=target_tensor[:, :-1] if target_tensor is not None else None
        )
        
        # Calculate loss with masking
        loss = masked_cross_entropy(
            decoder_outputs, 
            target_tensor[:, 1:],  # Shift targets
            devnagri2int[PAD_TOKEN]
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def masked_cross_entropy(logits, target, pad_idx):
    # logits: (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)
    mask = (target != pad_idx).float()
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target.reshape(-1)
    loss = F.nll_loss(logits_flat, target_flat, reduction='none')
    total_non_pad = mask.sum()
    loss = (loss * mask.view(-1)).sum() / (total_non_pad + 1e-6)
    return loss

def train(train_dataloader, val_dataloader,encoder, decoder, n_epochs, teacher_forcing_prob,beam_width,learning_rate=0.001,print_every=1, plot_every=100,iswandb=False):
    encoder.train() 
    decoder.train()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            # print(f"Word Validation Accuracy {evaluate_model(encoder,decoder,val_dataloader,int2devnagri,device,False)}")
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        if iswandb:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            wandb_runner.log({"train_loss": print_loss_avg,
                    #    "train_accuracy":evaluate_model(encoder=encoder,decoder=decoder,dataloader=train_dataloader,int2devnagri=int2devnagri,device=device,show_confusion=False,iswandb=iswandb,teacher_forcing_prob=teacher_forcing_prob,beam_width=beam_width),
                       "val_accuracy":evaluate_model(encoder=encoder,decoder=decoder,dataloader=val_dataloader,int2devnagri=int2devnagri,device=device,show_confusion=True,iswandb=iswandb,teacher_forcing_prob=teacher_forcing_prob,beam_width=beam_width), 
                       "epoch": epoch}
            )
    showPlot(plot_losses)