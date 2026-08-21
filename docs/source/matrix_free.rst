Matrix-free linear operators
============================

:class:`~koopman_graph.operators.LinearOperatorProtocol` is a common
``matvec`` / ``rmatvec`` / ``solve`` / ``expm_action`` / leading-eigpair
surface for networked maps that should not assemble
:math:`K_{\mathrm{eff}}`. It is **operator algebra**, not trainer DDP.

What ships
----------

Two teaching wrappers implement the protocol (off root ``__all__``):

* :class:`~koopman_graph.operators.MatrixFreeGraphLinearOperator` —
  one-tap :math:`I\otimes K_{\mathrm{self}}+\widehat{A}\otimes
  K_{\mathrm{nbr}}`, reusing the existing ``matrix_free`` helpers
* :class:`~koopman_graph.operators.PolynomialGraphLinearOperator` —
  monomial :math:`P\ge 2` graph polynomial
  :math:`\sum_{k=0}^{P}\widehat{A}^{k}\otimes K_{k}`

Dense assembly of a flat :math:`(N\cdot d)\times(N\cdot d)` matrix is
refused when :math:`N\cdot d` exceeds
:data:`~koopman_graph.operators.MAX_DENSE_LINEAR_OPERATOR_SIZE`
(4096; the same teaching ceiling as assembled eig-regularization /
joint Schur). :class:`~koopman_graph.operators.MemoryEstimate`
declares that cost without building the matrix. Trainer DDP and the
``[distributed]`` extras do **not** shrink the representation.

Leading eigpairs on this surface are Arnoldi Ritz values, not the
exact Kronecker reduction :math:`\operatorname{eig}(B(\lambda))`
with :math:`B(\lambda)=\sum_k\lambda^{k}K_{k}`. Kronecker exact
spectrum remains a special case on eligible
``koopman="graph"`` operators. :math:`10^{5}`-node scaling is not a
release gate.

How to use it
-------------

.. code-block:: python

   import torch
   from koopman_graph.operators import MatrixFreeGraphLinearOperator

   k_self = torch.tensor([[0.7, 0.05], [0.0, 0.6]], dtype=torch.float64)
   k_nbr = torch.tensor([[0.04, 0.0], [0.0, 0.03]], dtype=torch.float64)
   tails = torch.arange(5, dtype=torch.long)
   forward = torch.stack((tails, tails + 1), dim=0)
   edge_index = torch.cat((forward, forward.flip(0)), dim=1)
   op = MatrixFreeGraphLinearOperator(
       k_self, k_nbr, edge_index=edge_index, num_nodes=6
   )
   y = op.matvec(torch.randn(12, dtype=torch.float64))

Wrappers are **not** a factory kind. Default ``koopman=None`` still
selects ``"pernode"``.

Ceilings
--------

* Protocol ``@runtime_checkable`` checks method presence only.
* Dual / orbit layouts stay on dense assembly.
* Inverse of a :math:`P\neq 1` polynomial still uses Richardson on
  the polynomial map, not a cheap Kronecker inverse.
* This page does not quote a throughput number.

See :doc:`limitations` (Scale / matrix-free), :doc:`architecture`,
and :doc:`api`.
