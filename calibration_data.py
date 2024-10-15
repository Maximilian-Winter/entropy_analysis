# Collect calibration data
input_output_pairs = [
    (
        """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What's the capital of France?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        """
The capital of France is Paris.
<|eot_id|>""",
    ),
    (
        """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Explain the concept of photosynthesis.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        """
Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. It involves the following steps:

1. Light absorption by chlorophyll in plant leaves
2. Conversion of light energy to chemical energy (ATP and NADPH)
3. Use of this energy to convert CO2 and water into glucose
4. Release of oxygen as a byproduct

This process is crucial for life on Earth as it produces oxygen and forms the base of most food chains.
<|eot_id|>""",
    ),
    (
        """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Write a haiku about spring.
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        """
Here's a haiku about spring:

Soft breeze whispers by
Cherry blossoms paint the sky
New life awakens
<|eot_id|>""",
    ),
    (
        """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What are the three states of matter?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        """
The three main states of matter are:

1. Solid
2. Liquid
3. Gas

Each state has distinct properties based on the arrangement and movement of particles within the substance.
<|eot_id|>""",
    ),
    (
        """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 12 Oct 2024

You are an AI assistant created to be helpful and honest. Always think step by step and layout your chain of thought in great detail.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

How does a rainbow form?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        """
A rainbow forms through the following process:

1. Sunlight enters water droplets in the air
2. Light is refracted (bent) as it enters the droplet
3. The light is then reflected off the back of the droplet
4. As it exits, it's refracted again and separated into different wavelengths
5. This separation creates the spectrum of colors we see
6. The observer must be at the correct angle between the sun and water droplets to see the rainbow

Rainbows appear as an arc because of the specific angle at which this light refraction occurs.
<|eot_id|>""",
    ),
]