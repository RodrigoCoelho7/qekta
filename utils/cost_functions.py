import numpy as np
import torch

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

def frobenian_kta(K,y):
    """
    Implements the KTA as defined in https://link.springer.com/content/pdf/10.1007/s10462-012-9369-4.pdf.

    Denominator is the Frobenian Norm of the Kernel matrix K times the number of training samples.
    """
    N = y.shape[0]
    assert K.shape == (N,N), "Shape of K must be (N,N)"

    yT = y.view(1,-1) #Transpose of y, shape (1,N)
    Ky = torch.matmul(K,y) # K*y, shape (N,)
    numerator = torch.matmul(yT,Ky) #yT * Ky, shape (1,1) which is a scalar
    denominator = N * torch.norm(K, p = "fro")
    result = numerator / denominator
    return result.squeeze()

def centered_kta(K,y):
    """
    Implements the centered KTA which is better for unbalanced datasets according to https://link.springer.com/content/pdf/10.1007/s10462-012-9369-4.pdf.

    IMPORTANT: Always requires that both classes are represented in the batch.
    """
    # Ensure that K has the correct shape
    N = y.shape[0]
    assert K.shape == (N,N), "Shape of K must be (N,N)"

    # First need to calculate C = I - 11^T/N
    # Let's start by implementing I and 1
    I = torch.eye(N)
    one_vector = torch.ones((N,1))
    one_outer = torch.matmul(one_vector,one_vector.T) # This implements 11^T
    C = I - one_outer/N

    # Now let's implement K_c = CkC and Kstar_c = CKstarC where Kstar = yyT
    y = y.view(-1,1)
    Kstar = torch.matmul(y,y.T)
    k_c = torch.matmul(C, torch.matmul(K,C))
    kstar_c = torch.matmul(C, torch.matmul(Kstar,C))

    numerator = torch.sum(k_c * kstar_c)
    denominator = torch.norm(k_c, p ="fro") * torch.norm(kstar_c, p = "fro")
    result = numerator / denominator
    return result.squeeze()


def kta(K, y):
    """
    Implements the KTA as defined in https://pennylane.ai/qml/demos/tutorial_kernels_module/.

    Denominator is the square root of the trace of the kernel matrix squared times the number of training samples
    """
    # Ensure that K has the correct shape
    N = y.shape[0]
    assert K.shape == (N,N), "Shape of K must be (N,N)"

    yT = y.view(1,-1) #Transpose of y, shape (1,N)
    Ky = torch.matmul(K,y) # K*y, shape (N,)
    yTKy = torch.matmul(yT,Ky) #yT * Ky, shape (1,1) which is a scalar

    K2 = torch.matmul(K,K) #K^2, shape (N,N)
    trace_K2 = torch.trace(K2)

    result = yTKy / (torch.sqrt(trace_K2)* N)

    return result.squeeze()

def kernel_polarization(K,y):
    """
    Kernel polarization from https://link.springer.com/content/pdf/10.1007/s10462-012-9369-4.pdf.

    It's basically the same thing as KTA, different only from a normalization factor.
    """
    N = y.shape[0]
    assert K.shape == (N,N), "Shape of K must be (N,N)"

    yT = y.view(1,-1) #Transpose of y, shape (1,N)
    Ky = torch.matmul(K,y) # K*y, shape (N,)
    yTKy = torch.matmul(yT,Ky) #yT * Ky, shape (1,1) which is a scalar

    return yTKy.squeeze()