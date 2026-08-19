TOOLS = [

    # ==========================================================
    # SEARCH WEB
    # ==========================================================

    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information. "
                "Use this when you need up-to-date information "
                "or need to discover relevant web pages."
            ),
            "parameters": {
                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search query. "
                            "Example: "
                            "'top programming languages 2025'"
                        )
                    },

                    "max_results": {
                        "type": "integer",
                        "description": (
                            "Number of search results to return."
                        ),
                        "minimum": 1,
                        "maximum": 10
                    }
                },

                "required": [
                    "query"
                ],

                "additionalProperties": False
            }
        }
    },


    # ==========================================================
    # OPEN URL
    # ==========================================================

    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open and read a web page from an HTTP or HTTPS URL. "
                "Use this when search results contain a useful URL "
                "and you need to inspect the actual page contents."
            ),
            "parameters": {
                "type": "object",

                "properties": {

                    "url": {
                        "type": "string",
                        "description": (
                            "The complete HTTP or HTTPS URL "
                            "to open and read."
                        )
                    }
                },

                "required": [
                    "url"
                ],

                "additionalProperties": False
            }
        }
    },


    # ==========================================================
    # CALCULATE
    # ==========================================================

    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform a mathematical calculation."
            ),
            "parameters": {
                "type": "object",

                "properties": {

                    "expression": {
                        "type": "string",
                        "description": (
                            "Mathematical expression to calculate."
                        )
                    }
                },

                "required": [
                    "expression"
                ],

                "additionalProperties": False
            }
        }
    }

]