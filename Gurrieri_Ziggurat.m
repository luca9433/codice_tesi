X = gurrieridataset(:,1)
Y = gurrieridataset(:,2)
k_max = 0;
k = inf*ones([1 n]);
for i = 1:n
    for h = [1:(i-1),(i+1):n]
        if X(h)<X(i) && Y(h)-X(h)-Y(i)+X(i)>0 && X(h)>2*X(i)-Y(i) 
            && Y(h)<=Y(i) && X(i)-X(h) < k(i) % Triangolo in basso
                    k(i) = X(i)-X(h);
        elseif Y(h)>Y(i) && Y(h)-X(h)-Y(i)+X(i)>=0 && Y(h)<2*Y(i)
   -X(i) && X(h)>=X(i) && Y(h)-Y(i) < k(i) % Triangolo in alto
                    k(i) = Y(h)-Y(i);
        elseif Y(h)>Y(i) && X(h)<X(i) && Y(h)<X(h) +2*(Y(i)-X(i))
 && X(i)- X(h) + Y(h)-Y(i) < k(i)
                    k(i) = X(i)- X(h) + Y(h)-Y(i);
        elseif Y(i)-X(i) < k(i)
                    k(i)=Y(i)-X(i); % Altrimenti muore contro l' altopiano
        end
    end
    if k(i) > k_max
        k_max = k(i);
        j = i;
    end
end
k(j) = inf; % Assegno inf al k piu' elevato