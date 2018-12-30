

from eip import PLC
test = PLC()
test.IPAddress = "172.16.2.161"
value = test.Read("Timer.ACC")
print (value)
#Write a value: test.Write("tagname", value)
ex: test.Write("python_Real", 2.0)
test.Close()
