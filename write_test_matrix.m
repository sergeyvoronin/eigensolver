clear all;
m = 12;
n = 12;
A = randn(m,n);
B = randn(m,n);
C = A + i*B; 
A = real(C + C'); % symmetric

fp = fopen('data/mat1.txt','w');

fprintf(fp,'%d\n', m);
fprintf(fp,'%d\n', n);
for j=1:n
	for i=1:m
		fprintf(fp,'%4.4f\n',A(i,j));
	end
end

fclose(fp);

save('data/mat1.mat','A');

fprintf('eigen vecs/vals of A:\n');
[V,S] = eig(A)

