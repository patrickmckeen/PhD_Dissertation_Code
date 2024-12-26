global N
global dt
%FYI: in ALTRO they use 50 for the cutoff for slewing 180
%test 
%B_gram = magnetic_grammian(B_ECI);
%tf_index = condition_cutoff_time(B_gram, 800);

function B_gram = magnetic_grammian(B_ECI)
    global N dt
    %this function takes in a magnetic field of size
    % Nx3 and returns the magnetic field gramian B^T*B
    %of size 3x3xN
    B_gram = zeros(3,3,N);
    B1 = B_ECI(1, 1);
    B2 = B_ECI(1, 2);
    B3 = B_ECI(1, 3);
    B_gram_1 = [0 -B3 B2;
             B3  0 -B1;
              -B2 B1 0];
    B_gram(:, :, 1) = B_gram_1*B_gram_1;
    for i=2:N
        B1 = B_ECI(i, 1);
        B2 = B_ECI(i, 2);
        B3 = B_ECI(i, 3);
        B_skew_i = [0 -B3 B2;
                   B3  0 -B1;
                   -B2 B1 0];
        B_gram_i = B_skew_i*B_skew_i*dt;
        B_gram(:, :, i) = B_gram(:, :, i-1) + B_gram_i;
    end
end

function tf_index = condition_cutoff_time(B_gram,cutoff)
    %this function takes in a 3x3xN array of magnetic field
    %grammian and spits out the time index at which the cutoff
    % is hit for the condition number
    global N
    B_gram_cond = zeros(1,N);
    %generate 2-norm condition number of the grammian
    for i = 1:N
        B_gram_i = B_gram(:, :, i);
        B_gram_cond(i) = cond(B_gram_i, 2);
    end
    %find first location where the condition of the grammian is below the
    %cutoff
    tf_index = 0;
    for i = 1:N
        if B_gram_cond(i) < cutoff && tf_index == 0
            tf_index = i;
        end
    end
end