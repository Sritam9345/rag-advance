def table_format(tables):
    
    table_text=""
    
    for table in tables:

                if not table:
                    continue

                header = table[0]

                table_text += "| " + " | ".join(
                    str(x).strip() if x else ""
                    for x in header
                ) + " |\n"

                table_text += "|" + "|".join(
                    "---" for _ in header
                ) + "|\n"

                for row in table[1:]:

                    table_text += "| " + " | ".join(
                        str(x).strip() if x else ""
                        for x in row
                    ) + " |\n"

                table_text += "\n"
    return table_text
            