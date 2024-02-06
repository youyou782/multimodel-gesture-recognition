with open('./RSS/LR/LR-1-2.bin', 'rb') as file:
    # 读取所有字节
    byte_data = file.read()

# 转换每个字节为十进制数值
decimal_data = [byte for byte in byte_data]

# 打印十进制数值
print(decimal_data)
print('\n')
#print(decimal_data[72000])
print(len(decimal_data))
