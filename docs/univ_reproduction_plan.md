# UNIV reproduction plan

PSMAF-Net starts from a clean reproduction of the original UNIV project before
adding downstream validation tasks.

## Reproduction order

1. Import the original UNIV source, dependency notes, configuration conventions,
   and checkpoint expectations into the repository boundary.
2. Reproduce the original UNIV behavior as the upstream baseline.
3. Add the PSMAF fusion module only after the UNIV baseline is runnable and its
   feature-export contract is clear.
4. Connect downstream detection and segmentation tasks after the upstream UNIV
   and PSMAF components are established.

## Downstream order

Detection and segmentation are downstream validation tasks. They should not
reshape the upstream design before UNIV reproduction is complete:

- Detection is attached through the UNIV/PSMAF feature export path.
- Segmentation is attached through the UNIV/PSMAF feature export path.
