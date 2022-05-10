import sounddevice as sd

print()

print("All available devices:")
print(sd.query_devices())
print()
print("All available input devices:")
print(sd.query_devices(kind="input"))
print("---------------------------------------------------------------------------------------")

print("Default Query: ", sd.query_devices(device="default", kind="input"))

print("Device 1 Query(IQaudio): ", sd.query_devices(device=1, kind="input"))

print("Device 4 Query(Pulse): ", sd.query_devices(device="Pulse", kind="input"))
sd.default.samplerate = 48000
print(sd.default.samplerate)
