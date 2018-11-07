#!/usr/bin/env python3


class MonoVO:
    def __init__(self):
        pass
    """
    1. Capture images I_t and I_t+1.
    2. Undostort images I_t and I_t+1.
    3. Detect features (ORB) I_t and track features I_t+1.
    4. Nister's 5-point + RANSAC to compute essential matrix.
    5. Estimate rotation (R) and translation (t) from essential matrix.
    6. Scale R and t with external source.
    """

def main():
    pass

if __name__ == "__main__":
    main()
