import tensorflow as tf

if __name__ == '__main__':
    model_path = "E:\Hankamp\saved_models\V0.01_simpleCNN_Spectro_SNR10_MSWC_keywords_blue_red_unknown"   #Personal Desktop
    #model_path = "E:\Hankamp\RaspberryPi Zero\V0.04_WhiteNoise10dB_WithallUnknown_speech_command_dataset" #Personal Laptop
    #model_path = "/home/pi/tensorflow_models/...." #Raspbery Pi
    print("Converting model, file path: ", model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    
    tflite_model = converter.convert()
    print("TFLite model made, saving...")

    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
        print("TF Lite version saved in: ", f)
