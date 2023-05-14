# The Unnecessarily Fast ReLU Project
![image](https://github.com/johneckberg/Bitwise-ReLU/assets/93222362/0e1251ae-57dc-4a25-8f94-976e96214945)


Ah, the Rectified Linear Unit (ReLU), the bread and butter of activation functions. Efficient? Check. Differentiable? Check. Perfectly functional in its current state? Of course. But who needs functional when we can make it unnecessarily faster using bitwise operations?

In this project, we take the most famous of activation functions and give it a completely unnecessary speed boost, just for the sheer pleasure of knowing that we could. 

## Bitwise ReLU

Our bitwise ReLU implementation uses bit manipulation on floating-point numbers, which, let's be honest, is a bit of a hack. But what's life without a little bit of mischief?

```python
def relu(input_float):
    int_representation = np.float32(input_float).view('int32')
    mask = (int_representation >> 31)
    masked_representation = int_representation & ~mask
    masked_representation = np.int32(masked_representation).view('float32')
    return masked_representation
```

## Speed Comparison

We've taken the liberty of comparing the performance of our bitwise ReLU with the PyTorch implementation. Here's a code snippet to demonstrate how we've done that:

```python
large_tensor = torch.randn(1000000)

bitwise_relu_time = timeit(lambda: bitwise_relu(large_tensor))
pytorch_relu_time = timeit(lambda: torch.nn.functional.relu(large_tensor))

print(f"Bitwise ReLU time: {bitwise_relu_time}")
print(f"PyTorch ReLU time: {pytorch_relu_time}")
```

The results? Well, we'll let you run the code and find out. But be warned, the thrill of potentially unnecessary speed improvements can be addictive.

## A Note on Practicality

While our bitwise ReLU implementation may very well be an amusing diversion, we must remind you that the standard way to implement ReLU in both numpy and PyTorch is to use the `np.maximum` or `torch.clamp_min` functions, respectively. They are reliable, well-tested, and do not rely on bit manipulation hacks.

But if you're in the mood for a bit of fun, go ahead and give our bitwise ReLU a whirl.
