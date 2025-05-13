import tensorflow as tf


class GRUAttentionModel(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads=8, dropout_rate=0.3):
        super(GRUAttentionModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.encoder_gru = tf.keras.layers.GRU(hidden_dim, return_sequences=True, return_state=True, dropout=dropout_rate)

        # Multi-Head Attention Layer
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)

        self.decoder_gru = tf.keras.layers.GRU(hidden_dim, return_sequences=True, return_state=True, dropout=dropout_rate)
        self.fc_out = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, encoder_input, decoder_input):
        # Embedding
        encoder_emb = self.embedding(encoder_input)
        decoder_emb = self.embedding(decoder_input)

        # Encoder
        encoder_output, encoder_state = self.encoder_gru(encoder_emb)

        # Multi-Head Attention
        attention_output = self.multi_head_attention(query=decoder_emb, key=encoder_output, value=encoder_output)

        # Concatenate Attention Output with Decoder Input
        decoder_input_combined = tf.concat([decoder_emb, attention_output], axis=-1)
        decoder_output, _ = self.decoder_gru(decoder_input_combined, initial_state=encoder_state)

        # Output Layer
        output = self.fc_out(decoder_output)
        return output
