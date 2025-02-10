import sys

# []
# print(sys.argv[1])

letter_codes = [(ord(x)) for x in sys.argv[1]]
letter_codes += [0] * (21 - len(letter_codes))

print(f"int[{len(letter_codes)}](", end="")
for code in letter_codes[:-1]:
	print(f"{code}, ", end="")
print(f"{letter_codes[-1]})")