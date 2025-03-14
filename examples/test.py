import block
a = block.gen_had_block(2)
b = a.mate()
A, F = block.block.calculate_attraction(a, b)
print(A, F)