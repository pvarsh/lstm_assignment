require 'nngraph'

-- Seed for nn.Linear weights and random input generators
torch.manualSeed(123)

-- Sizes
size1 = 4
size2 = size1
size3 = 6


-- Nodes for gModule
x1 = nn.Identity()()
x2 = nn.Identity()()
lin_node = nn.Linear(size3, size2)()
mul_node = nn.CMulTable()({x2,lin_node})
add_node = nn.CAddTable()({mul_node, x1})

-- gModule for exercise answer
answer_gmod = nn.gModule({x1, x2, lin_node}, {add_node})

-- Generate inputs
a = torch.rand(size1)
b = torch.rand(size2)
c = torch.rand(size3)

-- Forward propagate the inputs and print answer
print("Testing on some random tensors")
print("Answer using gModule:")
print(answer_gmod:forward({a,b,c}))

-- Calculate the same answer step by step
torch.manualSeed(123)
lin = nn.Linear(size3, size2)
lin_c = lin:forward(c)
cmul = nn.CMulTable()
cadd = nn.CAddTable()
lin_c_cmul_b = cmul:forward({lin_c, b})
ans = cadd:forward({lin_c_cmul_b, a})
print("Answer without gModule:")
print(ans)
