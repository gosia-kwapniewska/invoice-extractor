prompt = """
    Extract the following structured information from the provided document:\n\n
    1. Parties:\n
       - Include each party 
       - its role (e.g., exporter, consignee).\n
       - Location for each party: 
           - city 
           - country.\n\n
    2. Country Overview:\n
       - Country of origin: only use explicit information from the document.\n
       - Country of destination: only use explicit information from the document.\n
       - Transit country: only if explicitly stated.\n
       NOTE: Do NOT guess the country of origin or destination based on the location of any party or company. 
    If the country is not explicitly mentioned, leave the field as null.\n\n
    3. Commodity Details (a list):\n
       - Description of goods\n
       - HS Code (HSC)\n\n
    4. Transportation:
       - Means of transport
       - vessel number.\n\n
    Respond strictly in valid JSON. If any field is missing, set its value to null.
    Below you'll find the structure:
    
    'Parties': [
        {
            'PartyName',
            'Role',
            'Location' {
                'City'
                'Country'
            }
        },
        ],
    'CountryOverview': {
        'CountryOfOrigin'
        'CountryOfDestination'
        'TransitCountry'
    },
    'CommodityDetails': [
        {
            'DescriptionOfGoods'
            'HSCode'
        }
    ],
    'Transportation': {
        'MeansOfTransport'
        'VesselNumber'
    }

"""