You will need to copy the modules from the required modules folder over to each test case you run.

This repository contains the software, data, and analysis scripts associated with the paper “Matrix‑Product Unitaries on Quantum Hardware.” All of the experiments described in the paper are represented here in runnable form, including the extraction tests, projection and reconstruction procedures, baseline comparisons, multilayer MPU evaluations, and the full stress‑test pipeline.

The repository includes:
• The exact scripts used to generate the hardware runs described in the paper.
• The JSON files returned by the IBM Quantum hardware during those runs, including calibration snapshots and raw bitstring counts.
• The analysis scripts used to extract entropy, mutual information, correlation matrices, collapse geometry signatures, bond profiles, and classical statistical distances.
• The Qiskit‑based code used to synthesize MPU operators, decompose them into native IBM gate sets, and prepare the input states described in the manuscript.

Several scripts are designed to contact the IBM Quantum cloud service. All API keys have been removed for security reasons. Because of this, any script that attempts to open a Runtime session or submit a job to hardware will fail unless the user supplies their own IBM Quantum API key. These failures are expected. Some scripts include explicit stops or guards to prevent accidental submission; others will raise an error when they attempt to authenticate. Users who wish to run the full pipeline on hardware must insert their own API key and, if desired, add additional stops or checks.

Even without an IBM Quantum account, many parts of the pipeline can still be executed. The scripts perform extensive checks prior to sending circuits to hardware, including MPU extraction, projection, reconstruction, circuit synthesis, transpilation, and pre‑execution validation. These stages run entirely locally and allow users to inspect the internal structure of the operators and verify the behavior of the Qiskit compiler without requiring access to the IBM cloud.

This repository is intended to make all experiments in the paper fully transparent and reproducible. Every numerical value and distribution reported in the manuscript originates from the scripts and data stored here.

The software in this repository is provided strictly on an “as‑is” basis. No guarantees, assurances, or warranties of any kind are offered regarding correctness, reliability, performance, or suitability for any purpose. The code was developed for research associated with the paper “Matrix‑Product Unitaries on Quantum Hardware,” and is not intended or recommended for production use in any environment.

All scripts, tools, and data files are made available solely for transparency and reproducibility of the research results. Any use, modification, or execution of this software is entirely at the user’s own risk. The author assumes no responsibility for errors, data loss, hardware interaction issues, failed executions, or any other consequences arising from the use of this repository.

Some scripts are designed to interact with IBM’s cloud‑based quantum computing services. All API keys have been removed. As a result, certain scripts will fail, raise exceptions, or terminate early unless the user supplies their own valid IBM Quantum API key. Users who choose to insert their own credentials accept full responsibility for any actions performed by the software.

By using any part of this repository, you agree that the author is not liable for any damages, losses, or issues that may occur. You are solely responsible for understanding the behavior of the scripts, reviewing them before execution, and ensuring they are appropriate for your intended use.





