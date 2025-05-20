
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import wandb
import time
from utils import int2devnagri,devnagri2int, showPlot,timeSince
from model import EOS_TOKEN,SOS_TOKEN
import pandas as pd
import os
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

def create_spaced_attention_view(attention_data, words_per_line=3, num_words=12):
    """
    Create a wide-spaced attention view for the given attention data.
    """
    samples = random.sample(attention_data, min(num_words, len(attention_data)))
    
    # Create figure
    fig = go.Figure()
    
    # Spacing configuration
    char_spacing = 15
    word_spacing = 60  
    line_spacing = 8   
    
    num_lines = (len(samples) + words_per_line - 1) // words_per_line
    
    for line_idx in range(num_lines):
        line_samples = samples[line_idx*words_per_line : (line_idx+1)*words_per_line]
        
        for word_idx, (latin, true, pred, mat, inp_toks, out_toks) in enumerate(line_samples):
            # Process attention matrix
            mat = mat.detach().cpu().numpy() if torch.is_tensor(mat) else np.array(mat)
            mat = mat[1:len(out_toks)+1, 2:len(inp_toks)+2]
            mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-9)
            
            # Calculate x positions with increased spacing
            x_start = word_idx * (len(out_toks) * char_spacing + word_spacing)
            x_positions = [x_start + i*char_spacing for i in range(len(out_toks))]
            
            # Add each character with hover info
            for char_idx, (char, x_pos) in enumerate(zip(out_toks, x_positions)):
                top_indices = np.argsort(mat[char_idx])[-3:][::-1]
                top_words = [inp_toks[i] for i in top_indices]
                top_weights = [f"{mat[char_idx][i]:.2f}" for i in top_indices]
                
                hover_text = (f"<b>Character:</b> {char}<br><b>Word:</b> {pred}<br>" +
                            "<br>".join([f"Attended to '{word}': {weight}" 
                                       for word, weight in zip(top_words, top_weights)]))
                
                fig.add_trace(go.Scatter(
                    x=[x_pos],
                    y=[-line_idx*line_spacing],  # Increased line spacing
                    mode="text",
                    text=char,
                    hovertext=hover_text,
                    hoverinfo="text",
                    textfont=dict(
                        size=28,  # Larger font size
                        family="Courier New, monospace",
                        color="#2a3f5f"  # Dark blue color
                    ),
                    showlegend=False
                ))
    
    # Style the layout with more space
    fig.update_layout(
        title=dict(
            text=" ",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-10, words_per_line*(15*6 + 60)]  # Adjusted range for wider display
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-num_lines*line_spacing - 2, 2]
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Courier New",
            bordercolor="#c7c7c7"
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=180 + num_lines*120,  # Increased height
        width=150 + words_per_line*200,  # Increased width
        margin=dict(l=40, r=40, b=40, t=100)
    )
    
    # Add subtle separation between word groups
    for line_idx in range(num_lines):
        fig.add_shape(
            type="line",
            x0=-10, y0=-line_idx*line_spacing-3,
            x1=words_per_line*(15*6 + 60), y1=-line_idx*line_spacing-3,
            line=dict(color="#e1e1e1", width=1, dash="dot")
        )
    
    wandb.log({"wide_spaced_attention": wandb.Plotly(fig)})
    return fig
    # return fig

def evaluate_model(encoder, decoder, dataloader, int2devnagri, devnagri2int,
                   device, teacher_forcing_prob, beam_width,
                   show_confusion=True, iswandb=False,
                   best_config=False, attention=False):
    """
    Evaluate the model on the given dataloader."""

    # set Devanagari font
    # font_path = 'C:/Users/aksha/Downloads/Noto_Sans_Devanagari/NotoSansDevanagari-VariableFont_wdth,wght.ttf'
    # font_manager.fontManager.addfont(font_path)
    # plt.rcParams['font.family'] = 'Noto Sans Devanagari'

    encoder.eval()
    decoder.eval()

    correct_words = 0
    total_words = 0
    y_true, y_pred = [], []
    samples = []
    attn_data = []  # for attention: list of (latin_entry, true_str, pred_str, attention_matrix, input_tokens, output_tokens)

    with torch.no_grad():
        for latin_words, _, inputs, targets, input_lengths, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # encode
            encoder_outputs, encoder_hidden, encoder_cell = encoder(inputs, input_lengths)

            # decode
            decoder_outputs, decoder_hidden, decoder_cell, attn_weights = decoder(
                encoder_outputs, encoder_hidden, encoder_cell,
                teacher_forcing_prob=teacher_forcing_prob,
                beam_width=beam_width
            )
            # attn_weights: (batch, tgt_len, src_len) if attention=True, else None

            # get predicted indices
            if decoder_outputs.dtype == torch.long:
                predicted_indices = decoder_outputs
            else:
                predicted_indices = decoder_outputs.argmax(dim=-1)

            for i, (pred_seq, true_seq) in enumerate(zip(predicted_indices, targets)):
                # build pred_list and true_list (filter PAD)
                if pred_seq.dim() == 0:
                    pred_list = [pred_seq.item()] if pred_seq.item() != devnagri2int[PAD_TOKEN] else []
                else:
                    pred_list = [t.item() for t in pred_seq if t.item() != devnagri2int[PAD_TOKEN]]
                true_list = [t.item() for t in true_seq if t.item() != devnagri2int[PAD_TOKEN]]

                # build strings
                pred_str, true_str = [], []
                for idx in pred_list:
                    if idx == devnagri2int[EOS_TOKEN]:
                        break
                    if idx != devnagri2int[SOS_TOKEN]:
                        pred_str.append(int2devnagri[idx])
                for idx in true_list:
                    if idx == devnagri2int[EOS_TOKEN]:
                        break
                    if idx != devnagri2int[SOS_TOKEN]:
                        true_str.append(int2devnagri[idx])

                pred_str = ''.join(pred_str)
                true_str = ''.join(true_str)

                # latin input: tuple of chars -> string
                latin_entry = latin_words[i]
                if isinstance(latin_entry, tuple):
                    latin_entry = ''.join(latin_entry)

                # correctness
                is_correct = (pred_str == true_str)
                samples.append((latin_entry, true_str, pred_str, is_correct))

                # record attention if requested and exists
                if attention and attn_weights is not None:
                    # convert to numpy
                    attn_mat = attn_weights[i].cpu().numpy()  # shape (tgt_len, src_len)
                    # also save token lists for axes
                    input_tokens = list(latin_entry)
                    output_tokens = list(pred_str)
                    attn_data.append((latin_entry, true_str, pred_str, attn_mat, input_tokens, output_tokens))

                # accuracy counts
                if is_correct:
                    correct_words += 1
                total_words += 1

                # char‐level confusion
                min_len = min(len(pred_list), len(true_list))
                y_pred.extend(pred_list[:min_len])
                y_true.extend(true_list[:min_len])

    word_accuracy = correct_words / total_words if total_words > 0 else 0.0

    # --- Plot accuracy bar and confusion ---
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(['Correct', 'Incorrect'],
                   [correct_words, total_words - correct_words],
                   color=['#4CAF50', '#F44336'])
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h,
                 f'{h:,}\n({h/total_words:.1%})',
                 ha='center', va='bottom')
    ax1.set_title(f'Word Accuracy: {word_accuracy:.2%}\nTotal Words: {total_words:,}')
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    if show_confusion and y_true and y_pred:
        ax2 = fig.add_subplot(1, 2, 2)
        labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        step = max(1, len(labels)//20)
        disp_labels = [int2devnagri[l] if i%step==0 else ''
                       for i, l in enumerate(labels)]
        sns.heatmap(cm_norm, ax=ax2,
                    cmap='YlOrRd',
                    xticklabels=disp_labels,
                    yticklabels=disp_labels,
                    cbar_kws={'label':'Accuracy'})
        ax2.set_title('Char Prediction Patterns')
        ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    if iswandb:
        wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.show()
    plt.savefig("acc.png")

    # --- Save predictions CSV ---
    if attention:
        os.makedirs("predictions_attention", exist_ok=True)
        pd.DataFrame(samples, columns=["Latin Input","True Devnagari","Predicted","Correct"]) \
          .to_csv("predictions_attention/predictions_attention.csv", index=False)
        table = wandb.Table(columns=["Latin Input","True Devnagari","Predicted","Correct"])
        for row in random.sample(samples, min(10,len(samples))):
            table.add_data(*row[:-1], "✔" if row[3] else "✖")
        wandb.log({"sample_predictions": table})
    elif best_config and iswandb:
        os.makedirs("predictions_vanilla", exist_ok=True)
        pd.DataFrame(samples, columns=["Latin Input","True Devnagari","Predicted","Correct"]) \
          .to_csv("predictions_vanilla/predictions_vanilla.csv", index=False)
        # log 10 sample rows
        table = wandb.Table(columns=["Latin Input","True Devnagari","Predicted","Correct"])
        for row in random.sample(samples, min(10,len(samples))):
            table.add_data(*row[:-1], "✔" if row[3] else "✖")
        wandb.log({"sample_predictions": table})

    # --- Plot attention heatmaps in 3x3 grid ---
    if attention and attn_data:
        os.makedirs("attention_plots", exist_ok=True)
        fig2, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)

        for ax, (latin, true, pred, mat, inp_toks, out_toks) in zip(
            axes.flatten(), random.sample(attn_data, min(9, len(attn_data)))
        ):
            # Ensure attention matrix is a numpy array
            mat = mat.detach().cpu().numpy() if torch.is_tensor(mat) else np.array(mat)

            # Normalize each row (i.e., output timestep's attention over input)
            # mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-8)
            mat = mat[1:len(out_toks)+1, 2:len(inp_toks)+2]
            row_sums = mat.sum(axis=1, keepdims=True) + 1e-9  # avoid divide by zero
            mat = mat / row_sums
            sns.heatmap(mat, ax=ax, cmap='viridis',
            cbar=True, vmin=0, vmax=1,
            yticklabels=out_toks)

            # Flip only the x-axis labels (not data)
            # Keep original data but reverse labels
            ax.set_xticks(np.arange(len(inp_toks)))
            ax.set_xticklabels(inp_toks[::-1])  # Reverse labels only
            ax.xaxis.tick_top()  # Optional: Move labels to top if clearer
            ax.set_title(f"{latin} → {pred}")
            ax.set_xlabel("Input chars")
            ax.set_ylabel("Output chars")
            plt.setp(ax.get_xticklabels(), rotation=90)

        fig2.suptitle("Attention Heatmaps (3×3 sample)", fontsize=16)
        plt.savefig("attention_plots/attention_grid.png")

        if iswandb:
            wandb.log({"attention_grid": wandb.Image(fig2)})
            create_spaced_attention_view(attention_data=attn_data)
        plt.show()


    encoder.train()
    decoder.train()
    return word_accuracy


def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer):
    """Train the model for one epoch."""
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
    """Compute the masked cross-entropy loss."""
    # logits: (batch_size, seq_len, vocab_size)
    # target: (batch_size, seq_len)
    mask = (target != pad_idx).float()
    logits_flat = logits.view(-1, logits.size(-1))
    target_flat = target.reshape(-1)
    loss = F.nll_loss(logits_flat, target_flat, reduction='none')
    total_non_pad = mask.sum()
    loss = (loss * mask.view(-1)).sum() / (total_non_pad + 1e-6)
    return loss

def train_model(train_dataloader, val_dataloader,encoder, decoder, n_epochs, teacher_forcing_prob,beam_width,learning_rate=0.001,print_every=1, plot_every=100,iswandb=False,best_config = False,attention=False):
    """
    Train the model with the given parameters."""
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
            if not best_config:
                wandb.log({
                    "train_loss": print_loss_avg,
                    "val_accuracy": evaluate_model(
                        encoder=encoder,
                        decoder=decoder,
                        dataloader=val_dataloader,
                        int2devnagri=int2devnagri,
                        devnagri2int = devnagri2int,
                        device=device,
                        show_confusion=True,
                        iswandb=iswandb,
                        teacher_forcing_prob=teacher_forcing_prob,
                        beam_width=beam_width,
                        best_config=best_config,
                        attention=attention
                    ),
                    "epoch": epoch
                })
            else:
                wandb.log({
                    "train_loss": print_loss_avg,
                    "test_accuracy": evaluate_model(
                        encoder=encoder,
                        decoder=decoder,
                        dataloader=val_dataloader,
                        int2devnagri=int2devnagri,
                        devnagri2int = devnagri2int,
                        device=device,
                        show_confusion=True,
                        iswandb=iswandb,
                        teacher_forcing_prob=teacher_forcing_prob,
                        beam_width=beam_width,
                        best_config=best_config,
                        attention=attention
                    ),
                    "epoch": epoch
                })
    showPlot(plot_losses)