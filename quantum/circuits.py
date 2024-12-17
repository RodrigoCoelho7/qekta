import pennylane as qml
import torch

def layer_hwe(x, input_scaling_params, rotational_params, wires, first_layer, use_data_reuploading):
    """
    x: input (batch_size,num_features)
    input_scaling_params: vector of parameters (num_features)
    rotational_params:  vector of parameters (num_features*2)
    """
    if first_layer or use_data_reuploading:    
        for i, wire in enumerate(wires):
            qml.RX(input_scaling_params[i] * x[:,i], wires = [wire])
    
    for i, wire in enumerate(wires):
        qml.RY(rotational_params[i], wires = [wire])

    for i, wire in enumerate(wires):
        qml.RZ(rotational_params[i+len(wires)], wires = [wire])

    if len(wires) == 2:
        qml.broadcast(unitary=qml.CZ, pattern = "chain", wires = wires)
    else:
        qml.broadcast(unitary=qml.CZ, pattern = "ring", wires = wires)


def ansatz_hwe(x, weights, wires, layers, use_data_reuploading):
    first_layer = True
    for layer in range(layers):
        layer_hwe(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading)
        first_layer = False

def layer_paper_adjoint(x,input_scaling_params, rotational_params, wires, first_layer, use_data_reuploading, coher_input_scaling, coher_variational, use_depolarizing_noise, depolarizing_strength):
    # Implements layer from paper https://arxiv.org/pdf/2105.02276

    if len(wires) == 2:
        qml.CRZ(rotational_params[0] + coher_variational[0], wires = wires)
    else:
        qml.adjoint(qml.broadcast)(unitary = qml.CRZ, pattern = "ring", wires = wires, parameters=rotational_params[len(wires):] + coher_variational[len(wires):] * rotational_params[len(wires):])

    if use_depolarizing_noise:
        for i,wire in enumerate(wires):
            qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)

    if first_layer or use_data_reuploading:
        for i, wire in enumerate(wires):
            qml.adjoint(qml.RY)(rotational_params[i] + coher_variational[i] * rotational_params[i], wires = [wire])

            if use_depolarizing_noise:
                qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)
            
            qml.adjoint(qml.RZ)((input_scaling_params[i] + coher_input_scaling[i] * input_scaling_params[i]) * x[:,i], wires = [wire])

            if use_depolarizing_noise:
                qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)
    
    for i,wire in enumerate(wires):
        qml.adjoint(qml.Hadamard)(wires = wire)

        if use_depolarizing_noise:
            qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)


def layer_paper(x,input_scaling_params, rotational_params, wires, first_layer, use_data_reuploading, coher_input_scaling, coher_variational, use_depolarizing_noise, depolarizing_strength):
    # Implements layer from paper https://arxiv.org/pdf/2105.02276

    for i,wire in enumerate(wires):
        qml.Hadamard(wires = wire)

        if use_depolarizing_noise:
            qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)

    if first_layer or use_data_reuploading:
        for i, wire in enumerate(wires):
            qml.RZ((input_scaling_params[i] + coher_input_scaling[i]) * x[:,i], wires = [wire])

            if use_depolarizing_noise:
                qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)

            qml.RY(rotational_params[i] + coher_variational[i], wires = [wire])

            if use_depolarizing_noise:
                qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)

    if len(wires) == 2:
        qml.CRZ(rotational_params[0] + coher_variational[0], wires = wires)
    else:
        qml.broadcast(unitary = qml.CRZ, pattern = "ring", wires = wires, parameters=rotational_params[len(wires):] + coher_variational[len(wires):])

    if use_depolarizing_noise:
        for i,wire in enumerate(wires):
            qml.DepolarizingChannel(p = depolarizing_strength, wires = wire)
        

def ansatz_paper(x, weights, wires, layers, use_data_reuploading, coherent_noises, use_depolarizing_noise, depolarizing_strenght, adjoint = False):
    first_layer = True
    if not adjoint:
        for layer in range(layers):
            layer_paper(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading, coherent_noises["input_scaling"][layer], coherent_noises["variational"][layer], use_depolarizing_noise, depolarizing_strenght)
            first_layer = False
    else:
        for layer in range(layers-1,-1,-1):
            layer_paper_adjoint(x, weights["input_scaling"][layer], weights["variational"][layer], wires, first_layer, use_data_reuploading, coherent_noises["input_scaling"][layer], coherent_noises["variational"][layer], use_depolarizing_noise, depolarizing_strenght)
            first_layer = False

def layer_uqc(x,w,b,phi,wires):
    for i,wire in enumerate(wires):
        wx_plus_b = torch.einsum("i,bi->b",2*w[i],x) + 2 * b[i]
        qml.RZ(wx_plus_b, wires = wire)
        qml.RY(2 * phi[i], wires = wire)

    if len(wires) == 2:
        qml.broadcast(unitary=qml.CZ, pattern = "chain", wires = wires)
    else:
        qml.broadcast(unitary=qml.CZ, pattern = "ring", wires = wires)

def ansatz_uqc(x,weights,wires,layers):
    for layer in range(layers):
        layer_uqc(x, weights["w"][layer], weights["b"][layer], weights["phi"][layer], wires)

def kernel_hwe(x,y,weights,wires,layers,projector,use_data_reuploading):
    x = x.repeat(1, len(wires) // len(x[0]) + 1)[:, :len(wires)]
    y = y.repeat(1, len(wires) // len(y[0]) + 1)[:, :len(wires)]
    ansatz_hwe(x,weights,wires,layers,use_data_reuploading)
    qml.adjoint(ansatz_hwe)(y,weights,wires,layers,use_data_reuploading)
    return qml.expval(qml.Hermitian(projector, wires = wires))

def kernel_paper(x,y,weights,wires,layers,projector,use_data_reuploading, coherent_noises, use_depolarizing_noise, depolarizing_strength):
    x = x.repeat(1, len(wires) // len(x[0]) + 1)[:, :len(wires)]
    y = y.repeat(1, len(wires) // len(y[0]) + 1)[:, :len(wires)]
    ansatz_paper(x,weights,wires,layers,use_data_reuploading, coherent_noises, use_depolarizing_noise, depolarizing_strength, adjoint = False)
    if use_depolarizing_noise:
        ansatz_paper(y,weights,wires,layers,use_data_reuploading, coherent_noises, use_depolarizing_noise, depolarizing_strength, adjoint = True)
    else:
        qml.adjoint(ansatz_paper)(y,weights,wires,layers,use_data_reuploading, coherent_noises, use_depolarizing_noise, depolarizing_strength, adjoint = False)
    return qml.expval(qml.Hermitian(projector, wires = wires))

def kernel_uqc(x,y,weights,wires,layers,projector,use_data_reuploading):
    ansatz_uqc(x,weights,wires,layers)
    qml.adjoint(ansatz_uqc)(y,weights,wires,layers)
    return qml.expval(qml.Hermitian(projector, wires = wires))





