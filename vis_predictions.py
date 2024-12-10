import torch
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

def parse_notes_to_windows(data, song_ids, log=False):
    """
    Parse a tensor of shape (windows, 1, notes) into a dictionary keyed by note,
    with values being lists of windows where the note exists.
    """

    # print('data', data.shape)
    num_samples, windows, num_instruments, notes = data.shape

    all_data = []
    for i in range(num_samples):
        sample = data[i]
        song_id = song_ids[i]
        note_to_windows = {note: [] for note in range(notes)}

        # Iterate through windows and notes
        for window_idx in range(windows):
            for note_idx in range(notes):
                if note_idx == 0:
                    continue
                
                if sample[window_idx, 0, note_idx] == 1:  # Check for presence of note
                    if log:
                        print('found note!!!', i, window_idx, note_idx)
                    note_to_windows[note_idx].append(window_idx)

                    # print(note_to_windows[note_idx])

        note_to_windows['id'] = song_id

        # print(note_to_windows)
        all_data.append(note_to_windows)

    return all_data


def visualize_notes(predictions, targets, num_windows=16, low=25, high=90):
    """
    Visualize the notes being played in both predictions and targets.

    :param predictions: List of dictionaries mapping note to windows for predictions.
    :param targets: List of dictionaries mapping note to windows for targets.
    :param num_windows: Number of windows to display.
    :param num_notes: Number of notes to display (129 notes).
    """

    num_notes = high - low + 1
    
    # Initialize empty matrices for predictions and targets
    pred_matrix = np.zeros((num_notes, num_windows))
    target_matrix = np.zeros((num_notes, num_windows))
    
    # Fill the prediction matrix
    for note, windows in predictions.items():
        if note == 'id' or note < low or note > high:
            continue
    
        # if windows:
        #     print('note', note, windows)
        for window in windows:
            # print('note ')
            pred_matrix[note-low, window] = 1

    # Fill the target matrix
    # for target in targets:
    for note, windows in targets.items():
        # print(note, windows)
        if note == 'id' or note < low or note > high:
            continue
            
        # if windows:
            # print('targetnote', note, windows)
        for window in windows:
            target_matrix[note-low, window] = 1

    # print(target_matrix)
    # Plotting the data
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = plt.cm.Blues
    cmap_colors = cmap(np.arange(cmap.N))
    cmap_colors[0, -1] = 0  # Set the alpha of the first color (for 0 values) to 0
    transparent_cmap = ListedColormap(cmap_colors)


    # Plot predictions
    ax.imshow(pred_matrix, aspect='auto', cmap=transparent_cmap, alpha=0.6, label="Predictions")
    
    # Plot targets
    ax.imshow(target_matrix, aspect='auto', cmap='Reds', interpolation='nearest', alpha=0.6, label="Targets")
    
    # Set labels and titles
    ax.set_xlabel('Windows')
    ax.set_ylabel('Notes')
    ax.set_title('Predictions and Targets for Notes Across Windows')

    # Set y-ticks to be notes 0-128
    ax.set_yticks(np.arange(num_notes))
    ax.set_yticklabels(np.arange(low, low+num_notes))
    
    # Add colorbar for visualizing intensity
    cbar = ax.figure.colorbar(ax.images[0], ax=ax, orientation='vertical')
    cbar.set_label('Presence of Note')
    
    # Show the plot
    plt.legend(['Predictions', 'Targets'])
    plt.show()




if __name__ == "__main__":
    # Load the saved predictions and targets
    data = torch.load('./predictions/predictions_targets_2.pt', map_location=torch.device('cpu'))
    predictions = data['predictions'] 
    targets = data['targets']   
    song_ids = data['song_ids'] 

    predictions = predictions[:10]
    targets = targets[:10]
    song_ids = song_ids[:10]

    print('preds shape', predictions.shape)
    print('targets shape', targets.shape)
    print('songs id shape', song_ids.shape)

    # Parse predictions and targets
    predictions_list = parse_notes_to_windows(predictions, song_ids, False)
    print('parsing preds', len(predictions_list), predictions_list[0])

    print('parsing targets')
    target_list = parse_notes_to_windows(targets, song_ids, False)
    # print(target_list[3])

    for i in range(len(predictions_list)):
        p = predictions_list[i]
        t = target_list[i]
        # print('p', p)
        # print('t', t)

        visualize_notes(p, t)


# for i in range(len(predictions_list)):
#     print("============\n\n")
#     p = predictions_list[i]
#     t = target_list[i]

#     print(p)
#     print(t)

# Print example of parsed dictionaries
# print("Predictions dictionary example:", {k: v for k, v in predictions_dict.items() if v})
# print("Targets dictionary example:", {k: v for k, v in targets_dict.items() if v})