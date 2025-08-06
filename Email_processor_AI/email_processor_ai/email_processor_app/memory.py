from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

sqlite_conn = sqlite3.connect("checkpoint.sqlite",check_same_thread=False)

memory = SqliteSaver(sqlite_conn)