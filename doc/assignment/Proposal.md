---
stylesheet: style.css
body_class: markdown-body
css: |-
  .page-break { page-break-after: always; }
  .markdown-body { font-size: 11px; }
  .markdown-body pre > code { white-space: pre-wrap; }
pdf_options:
  format: A4
  margin: 20mm 15mm
  printBackground: false
---
<!-- # Machinelearning on Microcontrollers - Project -->

# Proposal: Home automation voice commands
The goal is to train a machine learning algorithm that recognizes simple voice commands with a fixed pattern:

`[Callsign], [item] [value]`

> Edison, kitchen lights off!
>
> Edison, bedroom lights on!
>
> Edison, home cinema on!

### Platform
ARM with STM32 IoT node

### Dataset
*Hey Snips* to start and then generate own with onboard microphones

### ML Algorithm
Research for simple architecture, or try custom

### Implementation
- MFCC using CMSIS-DSP libraries
- CubeAI to start, later use CMSIS-NN

### Special
Choose one or more:
 - Will use CubeAI and if time allows, write a custom program to translate the network to optimized C code
 - Use small datasets and see how many samples are necessary for simple voice commands
 - Reduce power consumtion by waking from sleep on microphone activity

### Procedure
1. Train simple model on Hey Snips dataset to get platform up and running. Includes
    1. Train on hey snips
    2. Implement microphone sampling
    3. Implement MFCCs on target
    4. Implement snips ML + MFCC on target using CubeAI
2. Record own dataset with hand full of commands
3. Make changes in ML architecture

### Student
Noah Hütter, [huettern@student.ethz.ch](mailto:huettern@student.ethz.ch)