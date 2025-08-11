Debug Log Cheatsheet

Legend:
	•	[DBG][intro] — startup/introduction
	•	[DBG][stt] — speech‑to‑text (transcription & parsing)
	•	[DBG][chat] — main reply call (new question)
	•	[DBG][cont] — continuation reply call (after “yes/continue”)
	•	[DBG][cont-gate] — routing logic that decides whether to continue or cancel
	•	[DBG][ctx] — continuation and normal turn state changes (active/topic/round/history)
	•	[DBG][META?] / [DBG][META] — meta commands (voice/name changes)
	•	[DBG][err] — future errors (reserved tag)

Where logs come from (function → tags)
	•	main()
	•	[stt] Transcript: … — after STT transcription
	•	[stt] Norm: … | active=… round=… hist=… — normalized transcript + current context
	•	[stt] Flags: affirm=… neg=… — how we interpret the utterance
	•	[cont-gate] … — if a continuation is pending, shows the decision path
	•	[ctx] set active=… topic=… round=… hist=… — whenever we set/clear continuation or normal turn state; history is preserved with a rolling limit of 20 messages
	•	[intro] … — model’s introduction played at startup
	•	get_chat_reply() (new questions)
	•	[chat] requesting primary reply — chat/completions request sent; hist_len now refers to total conversation history sent, not just continuation context
	•	[chat] finish_reason=… words=… — response summary
	•	[chat] switching to brief mode — we’ll send the short version ending with “Would you like me to continue?”; brief-mode truncation still works the same but now keeps history for natural follow-up questions
	•	get_continuation_reply() (on “yes/continue”)
	•	[cont] hist_len=… round=… topic_preview=… — what context we’re sending back
	•	[cont] sanitizer removed greeting/intro or [cont] no sanitize needed — strip any “Hello, my name is…”
	•	[cont] still_more=… words=… — whether it asked to continue again
	•	parse_and_handle_meta()
	•	[META?] raw='…' lower='…' — parsed meta command candidate
	•	[META] set voice to … / [META] set name to … — confirmed meta action

How to read a turn
	1.	User speaks → [stt] Transcript … / Norm … / Flags …
	•	If active=True and affirm=True → continuation path.
	•	If active=True and neg=True → cancels continuation.
	•	Else → normal chat path.
	2.	Continuation path
	•	[cont-gate] active=True neg=False affirm=True → we call get_continuation_reply()
	•	[cont] hist_len=X round=Y … → context we passed
	•	[cont] still_more=False → chain ended; state resets in [ctx] …
	•	After the turn, user and assistant messages are appended to history.
	3.	Normal chat path
	•	[chat] … → main answer
	•	If we switch to brief mode, you’ll see [chat] switching to brief mode and then continuation is activated (see [ctx]).
	•	After the turn, user and assistant messages are appended to history.
Conversation memory now applies to all turns, not just continuations; history is preserved for normal chats and follow-ups with a rolling limit of 20 messages.