import os
import pickle

# Define the root directory to search
root_dir = 'runs/olmo-1b-ft/grad-store'  # Replace with the root directory

# Walk through the directory to find all 'grad_store.pkl' files
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == 'grad_store.pkl':
            file_path = os.path.join(dirpath, filename)
            
            # Open the .pkl file with pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            data = data.cpu()

            save_path = os.path.join(dirpath, 'grad_store.cpu.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"Processed and saved: {save_path}")