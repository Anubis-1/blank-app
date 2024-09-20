import streamlit as st

# Title of the app
st.title('SUPREME Transfer Express Quote Calculator')

# Transfer pricing data based on categories and sizes
pricing = {
    "Small": {
        "dimensions": (100, 100),  # Max width and height for Small
        "prices": [(1, 5, 8.00), (6, 10, 4.50), (11, 25, 2.50), (26, 50, 1.75), (51, 200, 1.25), (201, 500, 1.00)]
    },
    "Medium": {
        "dimensions": (265, 150),  # Max width and height for Medium
        "prices": [(1, 5, 10.00), (6, 10, 6.50), (11, 25, 4.50), (26, 50, 3.75), (51, 200, 3.25), (201, 500, 3.00)]
    },
    "Large": {
        "dimensions": (250, 300),  # Max width and height for Large
        "prices": [(1, 5, 13.00), (6, 10, 9.50), (11, 25, 7.50), (26, 50, 6.75), (51, 200, 6.25), (201, 500, 6.00)]
    }
}

# User input for width and height
width = st.number_input('Enter width (mm)', min_value=1, step=1)
height = st.number_input('Enter height (mm)', min_value=1, step=1)

# Function to determine size category based on width and height
def get_size_category(width, height):
    if width <= 100 and height <= 100:
        return "Small"
    elif width <= 265 and height <= 150:
        return "Medium"
    elif width <= 250 and height <= 300:
        return "Large"
    else:
        return None

# Determine the size category
size_category = get_size_category(width, height)

if size_category:
    st.write(f'Size category: {size_category}')
    
    # User input for quantity
    quantity = st.number_input('Enter quantity', min_value=1, step=1)

    # Function to calculate price based on quantity
    def calculate_price(size_category, quantity):
        for min_qty, max_qty, price in pricing[size_category]["prices"]:
            if min_qty <= quantity <= max_qty:
                return price
        return None

    # Calculate and display the result
    price_per_unit = calculate_price(size_category, quantity)

    if price_per_unit:
        total_cost = price_per_unit * quantity
        st.write(f'Price per unit: ${price_per_unit:.2f}')
        st.write(f'Total cost: ${total_cost:.2f}')
    else:
        st.write("Quantity exceeds available pricing tiers.")
else:
    st.write("Dimensions exceed the available categories.")