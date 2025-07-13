PROMPT_TEMPLATE = (
    'Analyze the stance of the following tweet regarding COVID-19 vaccines. '
    
    'Classify the tweet into exactly one of these categories:\n'
    '- in-favor: The tweet expresses positive sentiment, support, or endorsement of COVID-19 vaccines\n'
    '- against: The tweet expresses negative sentiment, opposition, skepticism, or criticism of COVID-19 vaccines\n'
    '- neutral-or-unclear: The tweet is neutral, unclear, doesn\'t mention COVID-19 vaccines, or expresses balanced/uncertain views\n\n'
    
    'Guidelines:\n'
    '- Look for explicit mentions of COVID-19 vaccines, vaccination, or related terms\n'
    '- Consider the overall sentiment and intent of the message\n'
    '- When in doubt about the stance, classify as neutral-or-unclear\n'
    '- Do not force a classification if the tweet is ambiguous\n\n'

     'Examples:\n'
    'Tweet: "please get vaxxed"\n'
    'Answer: in-favor\n\n'
    
    'Tweet: "COVID vaccines are dangerous and cause side effects."\n'
    'Answer: against\n\n'
    
    'Tweet: "The weather is nice today."\n'
    'Answer: neutral-or-unclear\n\n'
    
    'Tweet: "Some people support vaccines, others don\'t. It\'s complicated."\n'
    'Answer: neutral-or-unclear\n\n'
    
    'Tweet: "{tweet}"\n\n'
    'Answer with exactly one word: in-favor, against, or neutral-or-unclear.'
)
