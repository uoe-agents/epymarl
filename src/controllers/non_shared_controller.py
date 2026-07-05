from controllers.basic_controller import BasicMAC


class NonSharedMAC(BasicMAC):
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)  # bav
