import math
import sys
from typing import List

# Constants
PDF_SIZE = 512  # Example value; set this to the appropriate size
GAMMA = 1.0     # Example value; set this to the appropriate gamma value

class PDF:
    def __init__(self, xLeft: List[float], pdf: List[float]):
        if len(xLeft) < 2:
            raise ValueError("xLeft must contain at least two elements to calculate pdfStep.")
        if len(pdf) != PDF_SIZE:
            raise ValueError(f"pdf must contain exactly {PDF_SIZE} elements.")
        self.xLeft = xLeft
        self.pdf = pdf

def _quant_and_sat_cost(pdf: PDF, bw: int, delta: float, offset: int) -> float:
    """
    Calculate the quantization and saturation cost based on the provided PDF and encoding parameters.

    Args:
        pdf (PDF): The probability density function with xLeft and pdf attributes.
        bw (int): Bit width for the fixed-point representation.
        delta (float): The step size for the fixed-point representation.
        offset (int): The offset for the fixed-point representation.

    Returns:
        float: The calculated SQNR cost, capped at the maximum double value.
    """
    # Calculate the smallest and largest representable floating point values
    min_val = delta * offset
    step_size = math.pow(2, bw) - 1
    max_val = delta * (offset + step_size)

    # Calculate the indices of the smallest and largest representable values
    pdf_start = pdf.xLeft[0]
    pdf_step = pdf.xLeft[1] - pdf.xLeft[0]
    
    min_ind = math.floor((min_val - pdf_start) / pdf_step)
    min_ind = max(0, min(min_ind, PDF_SIZE - 1))
    
    max_ind = math.floor((max_val - pdf_start) / pdf_step)
    max_ind = max(0, min(max_ind, PDF_SIZE - 1))

    print(f"{min_ind=}, {max_ind=}, {delta=}, {offset=}")
    print(f"{min_val=}, {pdf_start=} {pdf_step=}")
    print(f"{pdf.xLeft[min_ind]=}")
    print()
    # Calculate the saturation cost of the bottom part of the PDF
    sat_cost_bottom = 0.0
    min_val_middle_of_bucket = pdf_start + (min_ind * pdf_step) + (pdf_step / 2)
    # min_val_middle_of_bucket = pdf.xLeft[min_ind] + (pdf_step / 2)
    # print(f"{min_ind=}\t{min_val_middle_of_bucket=}\t")
    for i in range(min_ind):
        mid_val = pdf_start + i * pdf_step + (pdf_step / 2)
        # mid_val = pdf.xLeft[i] + (pdf_step / 2)
        sat_cost_bottom += pdf.pdf[i] * math.pow(mid_val - min_val_middle_of_bucket, 2)

    # Calculate the saturation cost of the top part of the PDF
    sat_cost_top = 0.0
    max_val_middle_of_bucket = pdf_start + (max_ind * pdf_step) + (pdf_step / 2)
    # max_val_middle_of_bucket = pdf.xLeft[max_ind] + (pdf_step / 2)
    for i in range(max_ind, PDF_SIZE):
        mid_val = pdf_start + i * pdf_step + (pdf_step / 2)
        # mid_val = pdf.xLeft[i] + (pdf_step / 2)
        sat_cost_top += pdf.pdf[i] * math.pow(mid_val - max_val_middle_of_bucket, 2)

    # Calculate the quantization cost in the representable range of the PDF
    quant_cost = 0.0
    for i in range(min_ind, max_ind):
        float_val = pdf_start + i * pdf_step + (pdf_step / 2)
        quantized = round(float_val / delta - offset)
        dequantized = delta * (quantized + offset)
        quant_cost += pdf.pdf[i] * math.pow(float_val - dequantized, 2)

    # Calculate the total cost as the sum of quantization and saturation costs
    sqnr = GAMMA * (sat_cost_bottom + sat_cost_top) + quant_cost

    # Return the minimum of sqnr and the maximum double value
    return min(sqnr, sys.float_info.max)

if __name__ == "__main__":
    # Example PDF data
    xLeft = [i * 0.1 for i in range(PDF_SIZE + 1)]  # Example bin boundaries
    pdf_values = [1.0 / PDF_SIZE for _ in range(PDF_SIZE)]  # Uniform PDF for simplicity
    example_pdf = PDF(xLeft, pdf_values)

    # Example parameters
    bit_width = 8
    delta_value = 0.1
    offset_value = 10

    # Calculate the quantization and saturation cost
    cost = _quant_and_sat_cost(example_pdf, bit_width, delta_value, offset_value)
    # print(f"Quantization and Saturation Cost: {cost}")

