# RAPTOR - Software for Multi-scale Reactive Molecular Dynamics Simulations

RAPTOR® is an Open-Source software package for molecular dynamics simulations of chemical reactions in condensed phase. It is developed by the research group of Prof. Gregory A. Voth at Department of Chemistry, The University of Chicago. RAPTOR® stands for Rapid Approach for Proton Transport and Other Reactions.

![image](https://software.rcc.uchicago.edu/raptor/files/PT.gif "Proton transport via Grotthuss shuttling mechanism (changes of elctronic structures)")

Chemical reactions are founded in quantum dynamics. This naturally suggests quantum mechanical simulation methodologies (e.g. ab initio MD or QM/MM), which explicitly simulate the electronic structure, as the ideal method to approach reactive systems. However, the computational expense of these methods strongly limits their applicability to anything but the smallest systems. This has led to the development of reactive molecular mechanics models, such as Multiscale Reactive Molecular Dynamics method (MS-RMD) developed by the Voth group, which faithfully emulate reactive electronic structure through dynamic bonding. Properly parameterized reactive methods such as MS-RMD have the ability to efficiently capture chemical reactivity at a fraction of the computational cost.

The MS-RMD methodology is implemented by the Voth group in RAPTOR® as a plug-in package to LAMMPS, which is a popular molecular dynamics code developed by the Sandia National Laboratory. Therefore, RAPTOR® is compatible with mosft of the features and functions of LAMMPS and can be used for large-scale simulations over hundreds CPU cores via MPI.



## How to obtain RAPTOR

RAPTOR® is a free software product for all non-profit purposes. It is released in the Source Code version only, and needs to be compiled within the LAMMPS software, which is also free. To obtain more information and installation instructions for LAMMPS, please visit here. Please note that the distrition and use of LAMMPS is currently under the terms of GNU Public License, version 2. Furthermore, the use of RAPTOR® needs to be under a Non-Profit Licensing Agreement with its development team.

Please read the Licensing Agreement, download the [PDF version](https://software.rcc.uchicago.edu/raptor/files/RAPTORLicenseAgreement.pdf) of it. If you agree to this license agreement, then please send it back to Prof. Voth by email gavoth@uchicago.edu. 

The newest release of RAPTOR® (2022.3) is compatible with LAMMPS in version [23Jun2022](https://software.rcc.uchicago.edu/raptor/download/lammps-23Jun2022.tar.gz).

## Please Cite

- S. Kaiser, Z. Yue, Y. Peng, T. Nguyen, S. Chen, D. Teng, and G. A. Voth, “Molecular Dynamics Simulation of Complex Reactivity with the Rapid Approach for Proton Transport and Other Reactions (RAPTOR) Software Package”, _J. Phys. Chem. B._ **128**, 4959 – 4974 (2024). 
- T. Yamashita, Y. Peng, C. Knight, and G. A. Voth, “Computationally Efficient Multiconfigurational Reactive Molecular Dynamics”, _J. Chem. Theory Comp._ **8**, 4863−4875 (2012). PMCID: PMC412084.
- Y. Peng, C. Knight, P. Blood, L. Crosby, and G. A. Voth, “Extending Parallel Scalability of LAMMPS and Multiscale Reactive Molecular Simulations”, XSEDE’12: _Proceedings of the 1st Conference of the Extreme Science and Engineering Discovery Environment: Bridging from the eXtreme to the Campus and Beyond_, Article No. 37 (ACM, New York, 2012).