 elif loss_type == "FSM":
      output_P = output[y == 1]
      output_N = output[y == -1]
      s_p = torch.std(output_P)
      s_n = torch.std(output_N)
      m_p = torch.mean(output_P)
      m_n = torch.mean(output_N)
      loss =  (s_p+s_n)/torch.square(m_p-m_n+1e-9)