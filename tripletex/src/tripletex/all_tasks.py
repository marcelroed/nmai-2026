from tripletex.log_analysis import get_all_task_texts, get_logs

print("\n".join(get_all_task_texts(get_logs())))
