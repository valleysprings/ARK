PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
GRAPH_FIELD_SEP = "<SEP>"

PROMPTS[
    "entity_extraction"
] = """Extract entities and relationships from the text.

Format:
("entity"{tuple_delimiter}<NAME>{tuple_delimiter}<TYPE>{tuple_delimiter}<description under 30 words>)
("relationship"{tuple_delimiter}<SRC>{tuple_delimiter}<TGT>{tuple_delimiter}<description under 30 words>{tuple_delimiter}<strength 1-10>)

Separate with {record_delimiter}. End with {completion_delimiter}.

Example:
Text: Apple Inc. announced that CEO Tim Cook will present the new iPhone at their Cupertino headquarters.
Output:
("entity"{tuple_delimiter}"APPLE INC."{tuple_delimiter}"organization"{tuple_delimiter}"Technology company announcing new product."){record_delimiter}
("entity"{tuple_delimiter}"TIM COOK"{tuple_delimiter}"person"{tuple_delimiter}"CEO of Apple Inc."){record_delimiter}
("entity"{tuple_delimiter}"IPHONE"{tuple_delimiter}"product"{tuple_delimiter}"New product to be presented."){record_delimiter}
("entity"{tuple_delimiter}"CUPERTINO"{tuple_delimiter}"location"{tuple_delimiter}"Location of Apple headquarters."){record_delimiter}
("relationship"{tuple_delimiter}"TIM COOK"{tuple_delimiter}"APPLE INC."{tuple_delimiter}"CEO of the company"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"APPLE INC."{tuple_delimiter}"IPHONE"{tuple_delimiter}"Company announcing the product"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"APPLE INC."{tuple_delimiter}"CUPERTINO"{tuple_delimiter}"Headquarters location"{tuple_delimiter}8){completion_delimiter}

Text: {input_text}
Output:
"""