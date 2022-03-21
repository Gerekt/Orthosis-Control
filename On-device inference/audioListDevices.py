#import pyaudio
import sounddevice as sd

#p = pyaudio.PyAudio()
#for ii in range(p.get_device_count()):
#    print(p.get_device_info_by_index(ii).get('name'))

print()
print()

print(sd.query_devices())
print(sd.query_devices(kind="input"))
print("---------------------------------------------------------------------------------------")
print("Default Query: ", sd.query_devices(device="default", kind="input"))
print("Device 1 Query(IQaudio): ", sd.query_devices(device=1, kind="input"))
print("Device 4 Query(Pulse): ", sd.query_devices(device="Pulse", kind="input"))
sd.default.samplerate = 48000
print(sd.default.samplerate)
#print("second SECOND querry: ", sd.query_devices(device=14, kind="input"))
#print(sd.DeviceList())
