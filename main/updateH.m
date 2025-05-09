function H = updateH(V, mu, P, D, X, Y1, lambda2, I, Z)
    A = zeros(size(P{1}, 2));
    C = zeros(size(P{1}, 2), size(X{1}, 2));
    for i = 1:V
        A = A + mu * (P{i}' * P{i});
        G = X{i} - P{i} * D{i} - E_x{i} + Y1{i} / mu;
        C = C + mu * (P{i}' * G);
    end
    B = 2 * lambda2 * (I - Z) * (I - Z)';
    H = lyap(A, B, C);
end
