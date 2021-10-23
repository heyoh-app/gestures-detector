# To run the inference test
- Install requirements:
```
pip install -r multitask_lightning/inference_test/requirements.txt
```
- Convert the checkpoint to JIT and CoreML models:  
```
python -m multitask_lightning.conversion.convert multitask_lightning/checkpoints/epoch=126_val_loss=0_374_val_mAP=0_363.ckpt
```
- Run the script with model path (either .pt or .mlmodel):   
```
python -m multitask_lightning.inference_test.test --model_path=model.mlmodel --camera_id=0 --debug=False
```
<br />

To run inference test on Ubuntu with `--debug=False` make sure that OpenGL is installed:
```
sudo apt-get update
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
```
