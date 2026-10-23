"""Structural invariants shared by checkpoint loading and training constraints."""


def verify_recurrent_contract(cell):
    # An allowed anatomical edge may currently have zero learned weight and
    # become active again. The immutable anatomical mask/signs define legality.
    weights = cell._effective_W().detach()
    if ((weights[~cell._synapse_mask.bool()] != 0).any()
            or (cell.preserve_signs and (weights*cell._synapse_sign < 0).any())):
        raise ValueError('Continuation changed graph support/signs')
