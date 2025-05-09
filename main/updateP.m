function P=updateP(Xv,HD)
G = HD';
Q = Xv';
W = G'*Q;
[U,~,V] = svd(W,'econ');                                                                                                                                                                                                                              

PT = U*V';
P = PT';

end