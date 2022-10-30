% rotation 
syms L H d a b c
H1 = [1 0 0 d*cos(a);0 1 0 d*sin(a);0 0 1 H;0 0 0 1];
R1 = [cos(c) -sin(c) 0 0;sin(c) cos(c) 0 0;0 0 1 0;0 0 0 1];
H2 = H1*R1;
D1 = [1 0 0 L*sin(b); 0 1 0 0; 0 0 1 -L*cos(b); 0 0 0 1];
H3 = H2*D1;
D2 =[1 0 0 L*cos(b+30); 0 1 0 0; 0 0 1 L*sin(b+30); 0 0 0 1];
H4 = H2*D2;
D3 = [1 0 0 L*cos(b+270); 0 1 0 0; 0 0 1 L*sin(b+270); 0 0 0 1];
H5 = H2*D3;


