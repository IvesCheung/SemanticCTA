import json

from typing import List, Dict


def get_prompt(version_str: str = "base_profilling", **args) -> str:
    current_globals = globals()
    func_name = f"build_prompt_v_{version_str}"
    return current_globals.get(func_name)(**args)


def build_prompt_v_base_profilling(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    Basic prompt constructor for data profiling (column explanation).
    Expected: Model reads the base64-encoded full CSV and column name list, then outputs a JSON string:
      {
        "<col_name>": "<column explanation>",
        ...
      }
    """
    # table_name = args.get("table_name", "")
    # headers = args.get("headers", [])
    # csv_b64 = args.get("csv_encoded", "")
    # max_rows_hint = args.get(
    #     "max_rows_hint", "Sampling allowed; no need to explain row by row.")

    prompt_lines = [
        "You are a data analysis and data dictionary assistant. Your task is column-level data profiling.",
        "Using the provided column list and full table CSV content, infer each column’s meaning and characteristics and output a JSON mapping.",
        "",
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Columns:",
        ", ".join(headers),
        "CSV:",
        csv_encoded,
        "",
        "[Requirements]",
        "1. Output must be a valid JSON object string: keys are column names; values are explanations.",
        # "2. Each explanation should include: inferred type; example values (up to 3 typical or frequent); distribution (unique count / total rows; numeric min/max; date range start/end; null presence); possible business meaning (reasonable guess).",
        "2. Each explanation should include: inferred type; possible business meaning (reasonable guess).",
        f"3. Row count hint: {max_rows_hint}",
        "4. Output ONLY the JSON. No extra text, no markdown, no comments.",
        "5. If meaning unclear, use 'Meaning unclear' but still provide type and statistics.",
        "6. Parse the CSV content; do not fabricate fields not in the column list.",
        "",
        "[Example Output (format only, data must reflect actual content)]",
        "{",
        '  "column_a": "Type: Integer; Meaning: business primary key",',
        # '  "column_a": "Type: Integer; Examples: [1,2,5]; Dist: min=1 max=99 unique=35/500 rows; Nulls: none; Meaning: business primary key",',
        '  "column_b": "Type: String; Meaning: city name"',
        # '  "column_b": "Type: String; Examples: [\'Beijing\',\'Shanghai\',\'Guangzhou\']; Dist: unique=3/500 rows; Nulls: ~2%; Meaning: city name"',
        "}",
        "",
        "Now output the final JSON only."
    ]
    return "\n".join(prompt_lines)


def build_prompt_v_single_column(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    251223 versioned basic prompt constructor for data profiling (column explanation).
    Expected: Model reads the base64-encoded full CSV and column name list, then outputs a JSON string:
      {
        "<col_name>": "<column explanation>",
      }
    """
    prompt_lines = [
        "You are a data analysis assistant. Your task is to read and understand the given table, and then generate a brief description for each column of the data it represents.",
        "",
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers),
        "Values:",
        csv_encoded,
        "",
        "[Requirements]",
        "1. Output must be a valid JSON object string, in the form of {<COLUMN_NAME>: <COLUMN_DESCRIPTION>}",
        "2. Each description you generate (<COLUMN_DESCRIPTION>) must be a sentence describing the semantics of the corresponding column.",
        "3. Output ONLY the JSON object. No extra text, no markdown, no comments.",
        "[Example Table(format only)]",
        "Table name: retail_sales",
        "Headers:",
        "Transaction_ID, Quantity, Unit_Price",
        "Values:"
        "TXN-10025, 2, 45.99",
        "TXN-10026, 1, 12.50",
        "TXN-10027, 5, 45.99",
        "[Example Output(format only)]",
        "{",
        '  "Transaction_ID": "A unique alphanumeric code identifying a specific sales event."',
        '  "Quantity": "The number of units of the product purchased in this transaction."',
        '  "Unit_Price": "The price of a single unit of the product in the local currency."',
        "}",
    ]
    return "\n".join(prompt_lines)


def build_prompt_v_multi_column(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    Prompt constructor for mixed single-column profiling and multi-column relationship analysis.
    Expected: Model reads the CSV and column list, then outputs a JSON string:
      {
        "col_a": "Single column meaning",
        "col_a, col_b": "Relationship description",
        "col_a, col_b, col_c": "Relationship description",
        ...
      }
    """
    prompt_lines = [
        "You are a data analysis assistant. Your task is to perform both column-level profiling and multi-column relationship analysis.",
        "Using the provided column list and full table CSV content, infer each column’s meaning and identify relationships between columns.",
        "",
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Columns:",
        ", ".join(headers),
        "CSV:",
        csv_encoded,
        "",
        "[Requirements]",
        "1. Output must be a valid JSON object string.",
        "2. Keys can be:",
        "   - A single column name (for individual column profiling).",
        "   - A comma-separated list of 2 or more column names (e.g., 'col1, col2' or 'col1, col2, col3') for relationships.",
        "3. Single-column values must include: inferred type and possible business meaning.",
        "4. Multi-column values must describe the relationship. Explicitly consider and report when applicable:",
        "   - Composite key",
        "   - Hierarchy",
        "   - Correlation",
        "   - Functional dependency",
        "   - Aggregation constraint (sum/roll-up across columns)",
        "5. Include all columns individually. For relationships, include only significant ones.",
        f"6. Row count hint: {max_rows_hint}",
        "7. Output ONLY the JSON. No extra text, no markdown, no comments.",
        "",
        "[Example Output (format only)]",
        "{",
        '  "order_id": "Type: Integer; Meaning: Unique identifier for orders",',
        '  "line_item_id": "Type: Integer; Meaning: Line item sequence number",',
        '  "country": "Type: String; Meaning: Country name",',
        '  "city": "Type: String; Meaning: City name",',
        '  "order_id, line_item_id": "Composite key: order_id + line_item_id uniquely identifies rows",',
        '  "country, city": "Hierarchy: city belongs to country",',
        '  "age, income": "Correlation: numeric correlation observed",',
        '  "zip_code, city": "Functional dependency: zip_code determines city",',
        '  "female_count, male_count, total_enrollment": "Aggregation constraint: female_count + male_count = total_enrollment"',
        "}",
        "",
        "Now output the final JSON only."
    ]
    return "\n".join(prompt_lines)


def build_prompt_v_table_only(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    Prompt constructor for table profiling/description.
    Expected: Model reads the CSV and column list, then outputs a JSON string:
      {
        "__table__": "Brief table description",
      }
    """
    prompt_lines = [
        "You are a data analysis assistant. Your task is to read and understand the given table, and then generate a brief description of the data this table represents.",
        "",
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers),
        "Values:",
        csv_encoded,
        "",
        "[Requirements]",
        "1. Output must be a valid JSON object string, in the form of {\"__table__\": <DESCRIPTION>}",
        "2. The description you generate (<DESCRIPTION>) must be a noun phrase.",
        "3. Output ONLY the JSON object. No extra text, no markdown, no comments.",
        "",
        "[Example Output (format only)]",
        "{",
        "  \"__table__\": \"Course enrollment by gender\""
        "}",
        ""
    ]
    return "\n".join(prompt_lines)


def build_prompt_v_rephrase_table(profile, **args) -> str:
    """
    Prompt constructor for table profiling/description.
    Expected: Model reads the CSV and column list, then outputs a JSON string:
      {
        "__table__": "Brief table description",
      }
    """
    table_name = args.get("table_name", "")
    headers = args.get("headers", [])
    csv_encoded = args.get("csv_encoded", "")
    max_rows_hint = args.get("max_rows_hint", "(not provided)")

    if isinstance(profile, (dict, list)):
        original_desc = json.dumps(profile, ensure_ascii=False)
    else:
        original_desc = str(profile) if profile is not None else ""

    prompt_lines = [
        "You are a data analysis assistant. Your task is to rewrite an EXISTING table description into a better, concise table description.",
        "Use the table content as evidence when available. If the original description is already good, you may keep it with minor improvements.",
        "",
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers) if headers else "(not provided)",
        "Values:",
        csv_encoded if csv_encoded else "(not provided)",
        "",
        "Original description (to be rewritten):",
        original_desc if original_desc else "(not provided)",
        "",
        "[Requirements]",
        "1. Output must be a valid JSON object string, exactly in the form of {\"__table__\": <NEW_DESCRIPTION>}.",
        "2. <NEW_DESCRIPTION> must be a short noun phrase (not a full sentence).",
        "3. The new description must reflect the table content and be consistent with the original description.",
        "4. Do NOT invent information not supported by the table content or the original description.",
        f"5. Row count hint: {max_rows_hint}",
        "6. Output ONLY the JSON object. No extra text, no markdown, no comments.",
        "",
        "[Example Output (format only)]",
        "{",
        "  \"__table__\": \"Course enrollment by gender\"",
        "}",
        "",
        "Now output the final JSON only."
    ]
    return "\n".join(prompt_lines)


def build_prompt_v_rephrase_columns(profile, **args) -> str:
    """Rewrite per-column descriptions while preserving type and structure.

    Input: a table (optional) and an original profiling JSON-like object (profile).
    Output: the SAME nested JSON structure as the input profile, but with rewritten
    descriptions for each column. Preserve `__type__` values and keys exactly.

    Expected output example (structure only):
      {
        "colA": {"__type__": "int64", "colA": "...new...", "__table__": "..."},
        "colB": {"__type__": "object", "colB": "...new...", "__table__": "..."}
      }
    """
    table_name = args.get("table_name", "")
    headers = args.get("headers", [])
    csv_encoded = args.get("csv_encoded", "")
    max_rows_hint = args.get("max_rows_hint", "(not provided)")

    if isinstance(profile, (dict, list)):
        original_profile = json.dumps(profile, ensure_ascii=False)
    else:
        original_profile = str(profile) if profile is not None else ""

    prompt_lines = [
        "You are a data analysis assistant. Your task is to rewrite the EXISTING per-column descriptions in a profiling JSON.",
        "You MUST preserve the original nested JSON structure and keep all `__type__` fields unchanged.",
        "Use the table content as evidence when available.",
        "",
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers) if headers else "(not provided)",
        "Values:",
        csv_encoded if csv_encoded else "(not provided)",
        "",
        "Original profiling JSON (to be rewritten):",
        original_profile if original_profile else "(not provided)",
        "",
        "[Rewrite Rules]",
        "1. Output MUST be a valid JSON object string, and it MUST have the same top-level keys as the original profiling JSON.",
        "2. For each column object, keep these keys exactly if they exist: `__type__`, `<COLUMN_NAME>`, `__table__`.",
        "3. Preserve `__type__` values exactly (do not change types).",
        "4. Rewrite ONLY the description string values for each `<COLUMN_NAME>` field to be clearer and more accurate.",
        "5. Rewrite `__table__` into a better, concise table description (a short noun phrase). Keep it consistent across all columns.",
        "6. Do NOT add new columns/keys. Do NOT remove any existing keys.",
        "7. Do NOT invent facts not supported by the table content or the original profiling JSON.",
        f"8. Row count hint: {max_rows_hint}",
        "9. Output ONLY the JSON object. No extra text, no markdown, no comments.",
        "",
        "[Example Output (format only)]",
        "{",
        "  \"month\": {\"__type__\": \"object\", \"month\": \"The month and year when the metric was recorded.\", \"__table__\": \"Monthly telecom KPIs\"},",
        "  \"telco\": {\"__type__\": \"object\", \"telco\": \"The telecommunications provider the record refers to.\", \"__table__\": \"Monthly telecom KPIs\"}",
        "}",
        "",
        "Now output the final JSON only."
    ]
    return "\n".join(prompt_lines)


def build_prompt_v_COT_multi_column(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    251229 versioned COT prompt constructor for data profiling (column and subject-column explanation).
    Expected: The model reads the complete CSV data, first identifies the subject column, analyzes the relationships between columns, and then generates column descriptions containing relationships and semantics
    Output format: JSON string, with each column's description containing semantic information and its relationship to the subject column (if any)
    """
    prompt_lines = [
        "You are an expert data analyst. Your task is to analyze the given table with a systematic approach:",
        "1. FIRST, identify the MAIN SUBJECT COLUMN (the primary entity or key column that other columns describe)",
        "2. SECOND, analyze relationships between the main subject column and other columns",
        "3. THIRD, generate comprehensive descriptions for EACH column, including:",
        "   - The column's intrinsic semantics (what data it represents)",
        "   - Its relationship to the main subject column (if applicable)",
        "",
        "[Analysis Instructions]",
        "Follow this thinking chain:",
        "STEP 1 - Identify Main Subject Column:",
        "   - Look for columns that identify entities (like ID, code, name)",
        "   - Consider the table's likely primary entity (customers, products, transactions, etc.)",
        "   - Choose ONE column as the main subject/primary entity column",
        "",
        "STEP 2 - Analyze Column Relationships:",
        "   - For EACH column, determine how it relates to the main subject column",
        "   - Relationship types may include:",
        "     + Attribute/Property (describes characteristic of the main subject)",
        "     + Identifier/Key (uniquely identifies the main subject)",
        "     + Foreign Key (references another entity related to main subject)",
        "     + Measurement/Value (quantitative data about main subject)",
        "     + Temporal (time-related data about main subject)",
        "     + Categorical (classification of main subject)",
        "     + No direct relationship (independent data)",
        "",
        "STEP 3 - Generate Column Descriptions:",
        "   - For EACH column, create a comprehensive description containing:",
        "     + A. The column's core semantic meaning",
        "     + B. Its relationship to the main subject column (if exists)",
        "     + C. Any additional context from the data values",
        "   - Format: Clear, concise English sentences",
        "",
        "[Output Requirements]",
        "1. Output MUST be a valid JSON object string",
        "2. JSON structure: {",
        '   "<COLUMN_NAME>": "<DESCRIPTION>",',
        '   ...',
        "   }",
        "3. Each DESCRIPTION should be a sentence or two that includes:",
        "   - What the column represents semantically",
        "   - How it relates to the main subject column (e.g., 'This attribute of the customer...', 'This measurement for the transaction...')",
        "   - If no clear relationship, just describe the column's semantics",
        "4. DO NOT include the analysis steps in the output",
        "5. Output ONLY the JSON object - no additional text, markdown, or comments",
        "",
        "[Example Chain-of-Thoughts]",
        "Table name: customer_orders",
        "Headers:"
        "order_id, customer_id, order_date, product_name, quantity, unit_price, total_amount",
        "Values:",
        "(not provided)",
        "Thinking Chain:",
        "1. Main Subject Column: order_id (uniquely identifies each transaction)",
        "2. Relationships:",
        "   - customer_id: Foreign key linking to customer entity",
        "   - order_date: Temporal attribute of the order",
        "   - product_name: Attribute of what was ordered",
        "   - quantity: Measurement attribute of the order",
        "   - unit_price: Attribute of the product in the order",
        "   - total_amount: Calculated attribute of the order",
        "",
        "[Example Output]",
        "{",
        '  "order_id": "A unique identifier for each sales transaction, serving as the primary key for tracking individual orders."',
        '  "customer_id": "An identifier linking to the customer who placed the order, representing the customer responsible for this transaction."',
        '  "order_date": "The date when the order was placed, indicating the timing of this sales transaction."',
        '  "product_name": "The name or description of the product purchased in this order, specifying what item was transacted."',
        '  "quantity": "The number of units of the product ordered in this transaction, measuring the volume of the purchase."',
        '  "unit_price": "The price per unit of the product at the time of this order, representing the item\'s cost basis for the transaction."',
        '  "total_amount": "The calculated total cost for this order (quantity × unit_price), showing the monetary value of the transaction."',
        "}",
        "",
        "[Your Task]",
        "Now analyze the provided table and generate the JSON output following the above instructions.",
        "Remember: Identify the main subject column first, then analyze relationships, then write descriptions."
        "[Input Information]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers),
        "Values:",
        csv_encoded if csv_encoded else "(not provided)",
    ]

    return "\n".join(prompt_lines)


def build_prompt_v_COT2_multi_column(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    251230-2 versioned COT prompt constructor for data profiling (column and subject-column explanation).
    Expected: The model reads the complete CSV data, first identifies the subject column, analyzes the relationships between columns, and then generates column descriptions containing relationships and semantics
    Output format: Chain of thought process + "---JSON_RESULT---" + JSON result
    """
    prompt_lines = [
        "You are an expert data analyst. Your task is to analyze the given table and provide column descriptions with relationships.",
        "IMPORTANT: Your output MUST follow this EXACT format:",
        "1. First, provide your reasoning and analysis steps in plain text",
        "2. Then, on a new line, write the exact string: ---JSON_RESULT---",
        "3. Finally, output ONLY the JSON object (no additional text)",
        "",
        "[Thinking Instructions]",
        "",
        "STEP 1 - Identify Main Subject Column:",
        "   - Look for columns that identify entities (like ID, code, name)",
        "   - Consider the table's likely primary entity (customers, products, transactions, etc.)",
        "   - Choose ONE column as the main subject/primary entity column",
        "",
        "STEP 2 - Analyze Column Relationships:",
        "   - For EACH column, determine how it relates to the main subject column",
        "   - Relationship types may include:",
        "     + Attribute/Property (describes characteristic of the main subject)",
        "     + Identifier/Key (uniquely identifies the main subject)",
        "     + Foreign Key (references another entity related to main subject)",
        "     + Measurement/Value (quantitative data about main subject)",
        "     + Temporal (time-related data about main subject)",
        "     + Categorical (classification of main subject)",
        "     + No direct relationship (independent data)",
        "",
        "STEP 3 - Generate Column Descriptions:",
        "   - For EACH column, create a comprehensive description containing:",
        "     + A. The column's core semantic meaning",
        "     + B. Its relationship to the main subject column (if exists)",
        "     + C. Any additional context from the data values",
        "   - Format: Clear, concise English sentences",
        "",
        "[Output Requirements]",
        "After your reasoning, output the exact string '---JSON_RESULT---' on a new line, then output:",
        "A JSON object where keys are column names and values are descriptions.",
        "Each description should include:",
        "   - The column's semantic meaning",
        "   - Its relationship to the main subject column (if applicable)",
        "   - Be concise but informative (1-2 sentences)",
        "",
        "[Example Analysis and Output Format]",
        "Example Table: customer_orders",
        "Headers: order_id, customer_id, order_date, product_name, quantity, unit_price, total_amount",
        "[Example Analysis]",
        "Let me analyze this table...",
        "",
        "STEP 1 - Main Subject Column:",
        "The main subject column appears to be 'order_id' because it uniquely identifies each transaction...",
        "",
        "STEP 2 - Column Relationships:",
        "- 'order_date': This is a temporal attribute of the order...",
        "- 'customer_id': This is a foreign key that links to the customer entity...",
        "- 'order_date': This is a temporal attribute of the order...",
        "- 'product_name': This is the name of the product of the order...",
        "- 'quantity': This is the number of units of the product ordered...",
        "- 'unit_price': This is the price per unit of the product of the order...",
        "- 'total_amount': This is the calculated total cost for this order (quantity × unit_price)...",
        "",
        "STEP 3 - Generate Column Descriptions:",
        '  "order_id": "A unique identifier for each sales transaction, serving as the primary key for tracking individual orders."',
        '  "customer_id": "An identifier linking to the customer who placed the order, representing the customer responsible for this transaction."',
        '  "order_date": "The date when the order was placed, indicating the timing of this sales transaction."',
        '  "product_name": "The name or description of the product purchased in this order, specifying what item was transacted."',
        '  "quantity": "The number of units of the product ordered in this transaction, measuring the volume of the purchase."',
        '  "unit_price": "The price per unit of the product at the time of this order, representing the item\'s cost basis for the transaction."',
        '  "total_amount": "The calculated total cost for this order (quantity × unit_price), showing the monetary value of the transaction."',
        "",
        "---JSON_RESULT---",
        "{",
        '  "order_id": "A unique identifier for each sales transaction, serving as the primary key for tracking individual orders."',
        '  "customer_id": "An identifier linking to the customer who placed the order, representing the customer responsible for this transaction."',
        '  "order_date": "The date when the order was placed, indicating the timing of this sales transaction."',
        '  "product_name": "The name or description of the product purchased in this order, specifying what item was transacted."',
        '  "quantity": "The number of units of the product ordered in this transaction, measuring the volume of the purchase."',
        '  "unit_price": "The price per unit of the product at the time of this order, representing the item\'s cost basis for the transaction."',
        '  "total_amount": "The calculated total cost for this order (quantity × unit_price), showing the monetary value of the transaction."',
        "}",
        "",
        "[Your Task]",
        "Now analyze the provided table and generate your output following the exact format above.",
        "Remember: Reasoning first, then '---JSON_RESULT---', then only the JSON object."
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers),
        "Values:",
        csv_encoded,
        "",
    ]

    return "\n".join(prompt_lines)


def build_prompt_v_COT3_decisive(table_name, headers, csv_encoded, max_rows_hint, **args) -> str:
    """
    260127_1 versioned COT prompt constructor for data profiling (column and subject-column explanation).
    Expected: The model reads the complete CSV data, first identifies the subject column, analyzes the relationships between columns, and then generates column descriptions containing relationships and semantics
    Output format: Chain of thought process + "---JSON_RESULT---" + JSON result
    """
    prompt_lines = [
        "You are an expert data analyst with strong inference capabilities. Your task is to analyze the given table and provide column descriptions with relationships.",
        "IMPORTANT: Your output MUST follow this EXACT format:",
        "1. First, provide your reasoning and analysis steps in plain text",
        "2. Then, on a new line, write the exact string: ---JSON_RESULT---",
        "3. Finally, output ONLY the JSON object (no additional text)",
        "",
        "[Thinking Instructions]",
        "",
        "STEP 1 - Identify Main Subject Column:",
        "   - Look for columns that identify entities (like ID, code, name)",
        "   - Consider the table's likely primary entity (customers, products, transactions, etc.)",
        "   - Choose ONE column as the main subject/primary entity column",
        "",
        "STEP 2 - Analyze Column Relationships:",
        "   - For EACH column, determine how it relates to the main subject column",
        "   - Relationship types may include:",
        "     + Attribute/Property (describes characteristic of the main subject)",
        "     + Identifier/Key (uniquely identifies the main subject)",
        "     + Foreign Key (references another entity related to main subject)",
        "     + Measurement/Value (quantitative data about main subject)",
        "     + Temporal (time-related data about main subject)",
        "     + Categorical (classification of main subject)",
        "",
        "STEP 3 - Generate Column Descriptions & Handle Ambiguity (CRITICAL):",
        "   - For EACH column, create a comprehensive description containing:",
        "     + A. The column's core semantic meaning",
        "     + B. Its relationship to the main subject column (if exists)",
        "   - HANDLING AMBIGUITY / UNNAMED COLUMNS:",
        "     + If a header is missing, generic (e.g., 'col1', 'Var_2'), or unclear:",
        "       YOU MUST INFER the meaning based on the data values alone.",
        "     + Analyze value patterns (e.g., is it a date? a currency? a percentage? a categorical status? a specific ID format?).",
        "     + Analyze context (e.g., if column A is 'Quantity' and B is 'Price', and C is ambiguous but contains values equal to A*B, then C is 'Total Amount').",
        "     + Make the BEST EDUCATED GUESS based on the context.",
        "   - NEGATIVE CONSTRAINTS:",
        "     + Do NOT describe a column as 'unnamed', 'unknown', 'no specific purpose', or 'ambiguous'.",
        "     + Do NOT say 'the meaning cannot be determined'. Always provide a likely definition.",
        "",
        "[Output Requirements]",
        "After your reasoning, output the exact string '---JSON_RESULT---' on a new line, then output:",
        "A JSON object where keys are column names and values are descriptions.",
        "Each description should include:",
        "   - The inferred semantic meaning (even if guessed)",
        "   - Its relationship to the main subject column",
        "   - Be concise but informative (1-2 sentences)",
        "",
        "[Example Analysis and Output Format]",
        "Example Table: logistics_data",
        "Headers: shipment_id, Unnamed: 1, origin_city, dest_city, col_5",
        "Values: ['SHP001', '2023-10-01', 'New York', 'London', '15.5 kg']",
        "[Example Analysis]",
        "Let me analyze this table...",
        "",
        "STEP 1 - Main Subject Column:",
        "The main subject is 'shipment_id'...",
        "",
        "STEP 2 - Column Relationships:",
        "- 'shipment_id': Identifier...",
        "- 'Unnamed: 1': Contains dates. Likely the shipment date...",
        "- 'col_5': Contains numeric values with 'kg'. This is a weight measurement...",
        "",
        "STEP 3 - Generate Column Descriptions:",
        '  "shipment_id": "Unique identifier for the shipment."',
        '  "Unnamed: 1": "Inferred to be the shipment date based on the YYYY-MM-DD format, representing when the process occurred."',
        '  "origin_city": "The starting location of the shipment."',
        '  "dest_city": "The final destination of the shipment."',
        '  "col_5": "Inferred to be the shipment weight based on the \'kg\' suffix, representing the physical mass of the cargo."',
        "",
        "---JSON_RESULT---",
        "{",
        '  "shipment_id": "Unique identifier for the shipment."',
        '  "Unnamed: 1": "Inferred to be the shipment date based on the YYYY-MM-DD format, representing when the process occurred."',
        '  "origin_city": "The starting location of the shipment."',
        '  "dest_city": "The final destination of the shipment."',
        '  "col_5": "Inferred to be the shipment weight based on the \'kg\' suffix, representing the physical mass of the cargo."',
        "}",
        "",
        "[Your Task]",
        "Now analyze the provided table and generate your output following the exact format above.",
        "Remember: If a column is ambiguous, analyze the values to infer its purpose. Do not return 'unknown'."
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers),
        "Values:",
        csv_encoded,
        "",
    ]

    prompt_lines = [
        "You are an expert data analyst. Your task is to analyze the given table and provide column descriptions with relationships.",
        "IMPORTANT: Your output MUST follow this EXACT format:",
        "1. First, provide your reasoning and analysis steps in plain text",
        "2. Then, on a new line, write the exact string: ---JSON_RESULT---",
        "3. Finally, output ONLY the JSON object (no additional text)",
        "",
        "[Thinking Instructions]",
        "",
        "STEP 1 - Identify Main Subject Column:",
        "   - Look for columns that identify entities (like ID, code, name)",
        "   - Consider the table's likely primary entity (customers, products, transactions, etc.)",
        "   - Choose ONE column as the main subject/primary entity column",
        "",
        "STEP 2 - Analyze Column Relationships:",
        "   - For EACH column, determine how it relates to the main subject column",
        "   - Relationship types may include:",
        "     + Attribute/Property (describes characteristic of the main subject)",
        "     + Identifier/Key (uniquely identifies the main subject)",
        "     + Foreign Key (references another entity related to main subject)",
        "     + Measurement/Value (quantitative data about main subject)",
        "     + Temporal (time-related data about main subject)",
        "     + Categorical (classification of main subject)",
        "     + No direct relationship (independent data)",
        "",
        "STEP 3 - Generate Column Descriptions:",
        "   - For EACH column, create a comprehensive description containing:",
        "     + A. The column's core semantic meaning",
        "     + B. Its relationship to the main subject column (if exists)",
        "     + C. Any additional context from the data values",
        "   - Format: Clear, concise English sentences",
        "",
        "[Output Requirements]",
        "After your reasoning, output the exact string '---JSON_RESULT---' on a new line, then output:",
        "A JSON object where keys are column names and values are descriptions.",
        "Each description should include:",
        "   - The column's semantic meaning",
        "   - Its relationship to the main subject column (if applicable)",
        "   - Be concise but informative (1-2 sentences)",
        "",
        "[Example Analysis and Output Format]",
        "Example Table: customer_orders",
        "Headers: order_id, customer_id, order_date, product_name, quantity, unit_price, total_amount",
        "[Example Analysis]",
        "Let me analyze this table...",
        "",
        "STEP 1 - Main Subject Column:",
        "The main subject column appears to be 'order_id' because it uniquely identifies each transaction...",
        "",
        "STEP 2 - Column Relationships:",
        "- 'order_date': This is a temporal attribute of the order...",
        "- 'customer_id': This is a foreign key that links to the customer entity...",
        "- 'order_date': This is a temporal attribute of the order...",
        "- 'product_name': This is the name of the product of the order...",
        "- 'quantity': This is the number of units of the product ordered...",
        "- 'unit_price': This is the price per unit of the product of the order...",
        "- 'total_amount': This is the calculated total cost for this order (quantity × unit_price)...",
        "",
        "STEP 3 - Generate Column Descriptions:",
        '  "order_id": "A unique identifier for each sales transaction, serving as the primary key for tracking individual orders."',
        '  "customer_id": "An identifier linking to the customer who placed the order, representing the customer responsible for this transaction."',
        '  "order_date": "The date when the order was placed, indicating the timing of this sales transaction."',
        '  "product_name": "The name or description of the product purchased in this order, specifying what item was transacted."',
        '  "quantity": "The number of units of the product ordered in this transaction, measuring the volume of the purchase."',
        '  "unit_price": "The price per unit of the product at the time of this order, representing the item\'s cost basis for the transaction."',
        '  "total_amount": "The calculated total cost for this order (quantity × unit_price), showing the monetary value of the transaction."',
        "",
        "---JSON_RESULT---",
        "{",
        '  "order_id": "A unique identifier for each sales transaction, serving as the primary key for tracking individual orders."',
        '  "customer_id": "An identifier linking to the customer who placed the order, representing the customer responsible for this transaction."',
        '  "order_date": "The date when the order was placed, indicating the timing of this sales transaction."',
        '  "product_name": "The name or description of the product purchased in this order, specifying what item was transacted."',
        '  "quantity": "The number of units of the product ordered in this transaction, measuring the volume of the purchase."',
        '  "unit_price": "The price per unit of the product at the time of this order, representing the item\'s cost basis for the transaction."',
        '  "total_amount": "The calculated total cost for this order (quantity × unit_price), showing the monetary value of the transaction."',
        "}",
        "",
        "[Your Task]",
        "Now analyze the provided table and generate your output following the exact format above.",
        "Remember: Reasoning first, then '---JSON_RESULT---', then only the JSON object."
        "[Input]",
        f"Table name: {table_name}" if table_name else "Table name: (not provided)",
        "Headers:",
        ", ".join(headers),
        "Values:",
        csv_encoded,
        "",
    ]

    return "\n".join(prompt_lines)
