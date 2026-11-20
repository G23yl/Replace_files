def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")
        for index, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or index + 1 == len(actions):
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_indices = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_indices))
        actions = model_client.call(func_name="get_action_batch", obs=env_indices)
        for index in range(len(actions[0])):
            TASK_ENV.take_action_batch([chunk[index] for chunk in actions], env_indices)
            if TASK_ENV.is_episode_end() or index + 1 == len(actions[0]):
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [i for i, env_idx in enumerate(env_indices) if env_idx in running]
            actions = [actions[i] for i in active]
            env_indices = [env_indices[i] for i in active]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_indices))
