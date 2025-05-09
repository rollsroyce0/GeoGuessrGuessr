import time
import base64

for i in range(10): 
    # wait for 1 second
    time.sleep(0.1)
    current_time = time.time()
    #print("Current time:", current_time)

    round_time = round(current_time, 2)
    #print("Rounded time:", round_time)

    # Convert to binary string of the integer part
    binary = bin(int(round(round_time)))[2:]
    #print("Original binary:", binary)

    # Convert binary string to bytes
    # Pad to multiple of 8, then group into bytes
    padding_len = (8 - len(binary) % 8) % 8
    padded_binary = '0' * padding_len + binary
    binary_bytes = bytes(int(padded_binary[i:i+8], 2) for i in range(0, len(padded_binary), 8))

    # Encode bytes to Base32 (only uppercase letters and digits)
    base32_string = base64.b32encode(binary_bytes).decode('ascii')
    # cut off the last = sign if it exists
    if base32_string.endswith('='):
        base32_string_easy = base32_string[:-1]
    print("Alphanumeric Base32 string:", base32_string_easy)

    # Decode Base32 back to bytes
    decoded_bytes = base64.b32decode(base32_string.encode('ascii'))

    # Convert bytes back to binary string
    decoded_binary = ''.join(format(byte, '08b') for byte in decoded_bytes)

    # Remove leading zero padding for comparison
    decoded_binary_stripped = decoded_binary.lstrip('0')
    binary_stripped = binary.lstrip('0')

    assert decoded_binary_stripped == binary_stripped, "Decoded binary does not match original"
    #print("Decoded binary matches original")

    # Convert back to integer
    decoded_integer = int(decoded_binary_stripped, 2)
    #print("Decoded integer:", decoded_integer)
