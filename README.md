# Whisper Interpretability
This python notebook was created as part of the [Interpretability Hackathon](https://itch.io/jam/interpretability) hosted by Alignment Jam in November 2022. It demonstrates the use of Whisper, an end-to-end speech recognition model, and investigates the behavior of its encoder and decoder layers.

## Setup
To run this notebook, you will need to install the following dependencies:

* transformers
* datasets
* torch
* matplotlib

Alternatively, you can open and run directly in Google Calab.

## Usage
To generate the figures and results discussed in the summary below, run the cells in the notebook in order.

## Motivation
Multi-modal models, like Whisper, which can process multiple types of input data (e.g. speech and text) present unique safety risks compared to traditional large language models (LLMs) that only process text data. One such risk is the behavior of "hallucination," where the model produces the same phrase repeatedly in a transcript or output text that is not present in the input data. This behavior has been observed in LLMs, but not to the same extent as in Whisper, where it is frequently observed. This work is highly impactful and important as it helps to identify and mitigate these potential risks.

"Logit lens" is a popular concpt in LLMs and it refers to the behavior observed when the embeddings at the output of intermediate layers in a GPT-style language model are transformed into the vocabulary space using a final linear layer. This process results in an output that still makes sense, suggesting that each intermediate layer fine-tunes and improves the predictions of the model. I decided to delve into whether this behaviour is seen in Whisper on eitehr the encoder or decoder and if there is any link to hallucinations.

## Summary
It was observed that when removing layers from the Whisper decoder, the output remained readable and mostly correct when up to 3 layers were removed from the total of 24. However, the output quickly became unreadable and incorrect when more than 3 layers were removed. This suggests that the Whisper decoder exhibits "logit lens" behavior, where each layer fine-tunes the predictions and improves the sequence and confidence of tokens.

On the other hand, when removing layers from the Whisper encoder, the output remained correct for a longer period of time. Up to 8 layers could be removed from the total of 24 without significantly degrading the output. However, when more than 10 layers were removed, the output began to gradually degrade in quality, with repetition and "junk" output appearing when more than 15 layers were removed. These issues are very similar to hallucinations suggesting that perhaps logit lens is a good way of inducing them. The magnitude of the logits and the predicted tokens for each layer also showed that the later encoder layers helped with the generation of correct tokens later in the sequence, possibly by encoding longer range dependencies in the audio structure.

In the rest of the [report](https://itch.io/jam/interpretability/rate/1789933) (which was rated top for novelty in the Hackathon) we discuss how the attention patterns change during hallucinations.
