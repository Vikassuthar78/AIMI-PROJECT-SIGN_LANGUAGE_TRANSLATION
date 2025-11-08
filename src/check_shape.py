import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path="../models/sign_model_int8.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details())
