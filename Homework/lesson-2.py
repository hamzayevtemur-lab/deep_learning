import math

class Value:
    def __init__(self,data, _children=(), _op="", label=""):
        self.data=float(data)
        
        self.grad=0.0
        
        self._backward=lambda: None
        self._prev=set(_children)
        
        self._op=_op
        
        self.label=label
        
    def __repr__(self):
        return f"Value (data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other=other if isinstance(other, Value) else Value(other)
        
        out=Value(
            self.data+other.data,
            (self, other),
            "+"
        )
        
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
            
        out._backward=_backward
        
        return out
    
    def __radd__(self, other):
        return self+other
    
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self+(-other)
    
    def __rsub__(self, other):
        return other+(-self)
    
    def __mul__(self, other):
        other=other if isinstance(other, Value) else Value(other)
        
        out=Value(
            self.data * other.data,
            (self, other), "*"
        )
        
        def _backward():
            #d(a*b)/db=b
            self.grad+=other.data*out.grad
            
            # d(a*b)/db=a
            other.grad+=self.data*out.grad
            
        out._backward=_backward
        
        return out
    
    def __rmul__(self, other):
        return self*other
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        
        out=Value(
            self.data**power,
            (self,),
            f"**{power}"
        )
        
        def _backward():
            self.grad+=(
                power
                *(self.data**(power-1))
                *out.grad
            )
            
        out._backward=_backward
        
        return out
    
    def __truediv__(self, other):
        other=other if isinstance(other, Value) else Value(other)
        
        return self* (other**-1)

    def __rtruediv__(self, other):
        other=other if isinstance(other, Value) else Value(other)
        
        return other / self
    
    
    def relu(self):
        out=Value(
            max(0, self.data),
            (self,),
            "Relu"
        )
        
        def _backward():
            self.grad+=(out.data>0) * out.grad
            
        out._backward=_backward
        
        return out
    
    def sigmoid(self):
        s=1/(1+math.exp(-self.data))
        
        out=Value(
            s, (self,), "Sigmoid"
        )
        
        def _backward():
            self.grad+=s*(1-s)*out.grad
            
        out._backward=_backward
        
        return out
    
    def tanh(self):
        t=math.tanh(self.data)
        
        out=Value(
            t, (self, ),
            "Tanh"
        )
        
        def _backward():
            self.grad+=(1-t**2)*out.grad
            
        out._backward=_backward
        
        return out
    
    
    def backward(self):
        topo=[]
        visited=set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                 
                for child in v._prev:
                    build_topo(child)
                    
                topo.append(v)
                
        build_topo(self)
        
        self.grad=1.0
        
        for node in reversed(topo):
            node._backward()
            
            

### Testn 1

print("Example : y = x^2")

x = Value(5)

y = x ** 2

y.backward()

manual = 2 * 5

print(f"Forward Result : {y.data}")
print(f"Computed Gradient : {x.grad}")
print(f"Manual Gradient   : {manual}")
print(f"Match : {abs(x.grad - manual) < 1e-6}")

### Test 2

print("Example: y = x^2 + 3x + 1")

x = Value(2)

y = x**2 + 3*x + 1

y.backward()

manual = 2*2 + 3

print(f"Forward Result : {y.data}")
print(f"Computed Gradient : {x.grad}")
print(f"Manual Gradient   : {manual}")
print(f"Match : {abs(x.grad - manual) < 1e-6}")

    
### Test 3
print("Example: Sigmoid")

x = Value(0)

y = x.sigmoid()

y.backward()

manual = 0.25

print("Forward :", y.data)
print("Computed Gradient :", x.grad)
print("Manual Gradient   :", manual)
    
### Test 4
print("Example: Tanh")

x = Value(0)

y = x.tanh()

y.backward()

manual = 1

print("Forward :", y.data)
print("Computed Gradient :", x.grad)
print("Manual Gradient   :", manual)
    
    