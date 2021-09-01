[X,Y] = meshgrid(-5:0.01:5,-5:0.01:5);
x = sin(X)
y = sin(Y)
Z = x + y
mesh(X,Y,Z)