from datetime import datetime
import base64

def write_hyperparams_to_file(hyperparams, output_file):
    with open(output_file, 'a') as md:
        md.write('# Training Log Created at ' + str(datetime.now()) + '\n\n')

        # Hyperparameters table
        md.write('## Hyperparameters\n\n')
        md.write('| Parameter | Value |\n')
        md.write('|-----------|-------|\n')
        for k, v in hyperparams.items():
            md.write(f'| {k} | {v} |\n')
        md.write('\n')

def write_training_time_to_file(training_time, output_file):
    with open(output_file, 'a') as md:
        md.write('## Training Time\n\n')
        md.write('Total training time: ' + str(training_time) + ' seconds\n\n')

def write_loss_to_file(img_path, output_file):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('ascii')
    with open(output_file, 'a') as md:
        md.write('## Loss Curves\n\n')
        md.write(f'![Discriminator & Generator losses over epochs](data:image/png;base64,{img_b64})\n')

def write_real_vs_fake_to_file(img_path, output_file):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('ascii')
    with open(output_file, 'a') as md:
        md.write('## Real vs Fake Comparison\n\n')
        md.write(f'![Real vs Fake comparison for all materials](data:image/png;base64,{img_b64})\n')

def write_gif_to_file(img_path, output_file):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('ascii')
    with open(output_file, 'a') as md:
        md.write('## Synthesis Spectra during Training\n\n')
        md.write(f'![Synthesis Spectra during Training](data:image/gif;base64,{img_b64})\n')

def write_FID_to_file(FID_dict, img_path, output_file):
    with open(img_path, 'rb') as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('ascii')
    with open(output_file, 'a') as md:
        md.write('## FID Scores\n\n')
        md.write('| Material | FID Score |\n')
        md.write('|----------|-----------|\n')
        for material, fid in FID_dict.items():
            md.write(f'| {material} | {fid} |\n')
        md.write(f'![FID scores](data:image/gif;base64,{img_b64})\n')