@echo off

set "model_info_GP=_k4_GP"
set "model_info_CR=_k4_CR"
set "CUBLAS_WORKSPACE_CONFIG=:4096:8"

python "conditional_WGAN_2labels_train.py" ^
    --model_info %model_info_GP% --islog True --lr_D 3e-5 --lr_G 1e-4 --gp_coeff 10 --batch_size 50 --epoch_num 2 || exit /b
python "visualisation.py" ^
    --model_info %model_info_GP% --islog True || exit /b
python "generate_synthetic_data.py" ^
    --model_info %model_info_GP% || exit /b
python "spectra_FID.py" ^
    --model_info %model_info_GP% --islog True || exit /b
python "Vendi_score.py" ^
    --model_info %model_info_GP% --islog True || exit /b

python "conditional_WGAN_CR_train.py" ^
    --model_info %model_info_CR% --islog True --lr_D 3e-5 --lr_G 1e-4 --gp_coeff 10 --cl_coeff 10 --batch_size 50 --epoch_num 2 || exit /b
python "visualisation.py" ^
    --model_info %model_info_CR% --islog True || exit /b
python "generate_synthetic_data.py" ^
    --model_info %model_info_CR% || exit /b
python "spectra_FID.py" ^
    --model_info %model_info_CR% --islog True || exit /b
python "Vendi_score.py" ^
    --model_info %model_info_CR% --islog True || exit /b
