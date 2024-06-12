def test_alg_config_supports_reward(args):
    """
    Check whether algorithm supports specified reward configuration
    """
    if args.common_reward:
        # all algorithms support common reward
        return True
    else:
        if args.learner == "coma_learner" or args.learner == "qtran_learner":
            # COMA and QTRAN only support common reward
            return False
        elif args.learner == "q_learner" and (
            args.mixer == "vdn" or args.mixer == "qmix"
        ):
            # VDN and QMIX only support common reward
            return False
        else:
            return True
