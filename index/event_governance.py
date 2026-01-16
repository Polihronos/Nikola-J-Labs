class EventGovernance:
    def __init__(self):
        self.excluded_events = ["AIRDROP", "STAKING_REWARD", "EMISSIONS", "LOCK_UNLOCK"]

    def process_network_event(self, event_type, asset):
        """
        Determines the value contribution of a network event to the index.
        Returns 0 for excluded 'Agency' events.
        """
        # Strategy: Agency Neutrality.
        # We ignore staking and airdrops in the index level calculation.
        if event_type in self.excluded_events:
            print(f"[Governance] Event '{event_type}' for {asset} excluded by Agency Policy.")
            return 0 
            
        if event_type == "HARD_FORK":
            # Strategy: Consensus Inheritance.
            return self.resolve_fork(asset)
            
        return None # Just a signal that it's not excluded, or handle specific logic

    def resolve_fork(self, asset):
        """
        In a fork, we follow the chain with highest market cap / exchange consensus.
        """
        # In a real system, this would check live data. 
        # For this implementation, we return the 'Original' status assumption.
        print(f"[Governance] Resolving Hard Fork for {asset} -> Follow Market Cap Consensus.")
        return "FOLLOW_CONSENSUS"
